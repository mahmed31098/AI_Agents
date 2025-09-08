# main.py
"""
Web Search Agent - main.py
Implements:
 - secure .env handling for SEARCH_API_KEY
 - Async web-search tool (example uses Tavily-like API)
 - Dynamic (callable) system prompt that personalises per-user
 - Keyword-driven dynamic instruction mutation (deeper / summarise / links)
 - Streaming output via Runner.run_streamed; fallback to synchronous run
 - Simple in-memory caching with TTL to reduce repeated API calls
 - Clear logging and error handling

Adjust:
 - SEARCH_API_URL to your chosen search API endpoint
 - Model, temperature, max_tokens in llm_model configuration
 - Any library imports if your environment uses different names for the Agents SDK
"""

import os
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from functools import wraps

import httpx
from dotenv import load_dotenv, find_dotenv

# The SDK imports (these names align with your earlier example).
# If your environment uses other names, adjust accordingly.
from openai.types.responses import ResponseTextDeltaEvent

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    RunContextWrapper,
)

# ----------------------------
# Configuration & Logging
# ----------------------------
load_dotenv(find_dotenv())

SEARCH_API_KEY: Optional[str] = os.environ.get("GOOGLE_API_KEY")
if not SEARCH_API_KEY:
    raise RuntimeError("SEARCH_API_KEY missing. Add it to a .env file in the project root.")

# Example search API URL. Replace with your chosen API (Tavily/Bing/Google Custom Search).
SEARCH_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# LLM & agent tuning defaults - change as needed
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 350

# Search defaults
DEFAULT_NUM_RESULTS = 5
SEARCH_TIMEOUT = 10.0  # seconds

# Caching (simple in-memory TTL)
CACHE_TTL_SECONDS = 60 * 5  # 5 minutes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("web_search_agent")

# ----------------------------
# Simple cache implementation
# ----------------------------
class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self.ttl:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any):
        self._store[key] = {"value": value, "ts": time.time()}

cache = SimpleTTLCache()

# ----------------------------
# User context dataclass
# ----------------------------
@dataclass
class UserProfile:
    name: str
    city: Optional[str] = None
    topic: Optional[str] = None

# ----------------------------
# Initialize LLM client + model
# ----------------------------
set_tracing_disabled(disabled=True)  # keep SDK tracing quiet; remove for debug

# Create an external client instance for the SDK (example uses AsyncOpenAI wrapper).
external_client = AsyncOpenAI(api_key=SEARCH_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

llm_model = OpenAIChatCompletionsModel(
    model=LLM_MODEL_NAME,
    openai_client=external_client
)

# ----------------------------
# Utility helpers
# ----------------------------
def cache_key_for_query(query: str, num_results: int) -> str:
    return f"search:{num_results}:{query.strip().lower()}"

def safe_extract_results(api_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Normalize search API JSON into a list of {title, url, snippet}.
    This function may need to be adapted depending on the exact API response shape.
    """
    results = []
    raw = api_json.get("results") or api_json.get("items") or []
    for item in raw:
        title = item.get("title") or item.get("name") or item.get("headline") or ""
        url = item.get("url") or item.get("link") or item.get("display_url") or ""
        snippet = item.get("snippet") or item.get("description") or item.get("snippet_text") or ""
        results.append({"title": title, "url": url, "snippet": snippet})
    return results

# ----------------------------
# Search tool (callable by agent)
# ----------------------------
@function_tool
async def search_web(local_context: RunContextWrapper[UserProfile], query: str, num_results: int = DEFAULT_NUM_RESULTS) -> str:
    """
    Tool that the agent calls to do web searches.
    - Respects an in-memory cache (TTL)
    - Returns a formatted result (title + URL + snippet)
    - Access to local_context to personalise or to log the requesting user
    """
    user_profile = local_context.context if local_context else None
    logger.info("search_web called by user=%s query=%s", getattr(user_profile, "name", None), query)

    # Quick cache check
    key = cache_key_for_query(query, num_results)
    cached = cache.get(key)
    if cached:
        logger.debug("cache hit for query: %s", query)
        return cached

    headers = {"Authorization": f"Bearer {SEARCH_API_KEY}"}  # or API-specific header
    params = {"query": query, "num_results": num_results}

    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.get(SEARCH_API_URL, headers=headers, params=params)
            if resp.status_code != 200:
                logger.warning("Search API returned %s: %s", resp.status_code, resp.text[:200])
                return f"Search API error: {resp.status_code}"
            data = resp.json()
    except httpx.RequestError as e:
        logger.exception("Network error calling search API")
        return f"Network error while calling search API: {str(e)}"
    except Exception as e:
        logger.exception("Unexpected error calling search API")
        return f"Unexpected error while calling search API: {str(e)}"

    results = safe_extract_results(data)
    if not results:
        formatted = "No search results found."
        cache.set(key, formatted)
        return formatted

    # Format top results into a readable plain-text block
    formatted_lines = []
    for r in results:
        title = r.get("title", "<no title>")
        url = r.get("url", "<no url>")
        snippet = r.get("snippet", "").strip()
        formatted_lines.append(f"- {title} ({url})\n  {snippet}")

    formatted = "\n".join(formatted_lines)
    cache.set(key, formatted)
    return formatted

# ----------------------------
# Dynamic instructions generator
# ----------------------------
async def personalised_prompt(local_context: RunContextWrapper[UserProfile], agent: Agent[UserProfile]) -> str:
    """
    Return a dynamic system prompt that:
      - Personalises using the user's profile
      - Detects keywords in the last user input to mutate behaviour
    Expected signature: (local_context, agent) -> str
    """
    profile = local_context.context if local_context else None
    user_name = getattr(profile, "name", "User")
    city = getattr(profile, "city", None)
    topic = getattr(profile, "topic", None)

    # Base instruction
    instruction = [
        f"You are {agent.name}, a concise, helpful web-search assistant.",
        f"You're helping {user_name}" + (f" from {city}" if city else "") + (f" who likes {topic}" if topic else ""),
        "When producing answers: provide a brief 3-sentence summary followed by bullet-pointed links (title + url + short snippet).",
        "If the user asks for 'deeper' search, increase the number of results and provide more bullet points.",
        "If the user asks 'just links' or 'only links', return only bullet-pointed links (no prose).",
        "When possible, mention the most recent year or date if the query is time-sensitive."
    ]

    # Inspect last user message (if present in context.extra or local_context). The exact place depends on your Runner usage.
    # Some SDKs pass the latest user input in local_context.extra or via the Runner call - here we attempt to read commonly used fields.
    last_user_text = None
    if hasattr(local_context, "last_user_message"):
        last_user_text = getattr(local_context, "last_user_message")
    elif isinstance(local_context, dict) and local_context.get("last_user_message"):
        last_user_text = local_context.get("last_user_message")

    # Fallback: look for explicit instruction keywords in the agent object's last input (if available)
    # (This section is flexible depending on your SDK; it is safe to ignore if not available.)
    if last_user_text:
        lt = last_user_text.lower()
        if "deeper" in lt or "more results" in lt or "search deeper" in lt:
            instruction.append("User requested a deeper search: increase results and include extra citations.")
        if "just links" in lt or "only links" in lt or "give me links" in lt:
            instruction = [
                f"You are {agent.name}.",
                f"Return only bullet-pointed links (title and URL) — no summary or prose."
            ]

    return "\n".join(instruction)

# ----------------------------
# Create the agent
# ----------------------------
web_search_agent = Agent[UserProfile](
    name="WebSearchGenius",
    instructions=personalised_prompt,
    model=llm_model,
    tools=[search_web],
)

# ----------------------------
# Runner helpers - streaming & non-streaming
# ----------------------------
async def run_streamed_query(user_profile: UserProfile, user_input: str):
    """
    Run the agent in streaming mode and print textual deltas as they arrive.
    This depends on the Runner.run_streamed implementation in your SDK.
    """
    logger.info("Starting streamed run for user=%s input=%s", user_profile.name, user_input)

    # If your SDK expects extra metadata (like last user message) inside context,
    # you might want to create a wrapper. The SDK from your earlier code used RunContextWrapper when calling
    # the tools; here we pass the dataclass instance as context which the Agent's callable will receive.
    # If your SDK supports adding last_user_message into context, you could wrap it.
    try:
        stream = Runner.run_streamed(
            starting_agent=web_search_agent,
            input=user_input,
            context=user_profile
        )
    except TypeError:
        # Some SDKs may have different signatures for run_streamed; fallback to Runner.run (non-streaming).
        logger.warning("Runner.run_streamed signature mismatch; falling back to non-streamed run.")
        return await run_sync_query(user_profile, user_input)

    # Iterate events as they appear
    async for event in stream.stream_events():
        # event.type & event.data structure depends on SDK; your earlier example used:
        #  - event.type == "raw_response_event"
        #  - isinstance(event.data, ResponseTextDeltaEvent)
        try:
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                # event.data.delta is the incremental text
                print(event.data.delta, end="", flush=True)
            else:
                # Fallback to printing other event types for debugging and visibility
                # For example you may get "tool_call", "tool_response", "final_output" etc.
                data_text = getattr(event, "data", None)
                if data_text:
                    # print representation but keep it compact
                    print(str(data_text), end="", flush=True)
        except Exception as e:
            logger.exception("Error while processing stream event: %s", e)

    print()  # newline after streaming finishes

async def run_sync_query(user_profile: UserProfile, user_input: str):
    """
    Runs the agent synchronously (non-streaming) using Runner.run.
    Prints final output.
    """
    logger.info("Starting non-streamed run for user=%s input=%s", user_profile.name, user_input)
    try:
        result = await Runner.run(
            starting_agent=web_search_agent,
            input=user_input,
            context=user_profile
        )
    except Exception as e:
        logger.exception("Run failed")
        print(f"Agent run failed: {e}")
        return

    # result format depends on SDK; attempt to print common fields
    final_text = getattr(result, "final_output", None) or getattr(result, "output_text", None) or str(result)
    print(final_text)

# ----------------------------
# CLI entrypoint
# ----------------------------
def _cli_input_prompt() -> str:
    return input("\nAsk the WebSearch agent (type 'quit' to exit):\n> ").strip()

async def interactive_loop(default_profile: Optional[UserProfile] = None):
    """
    Simple interactive loop for testing the agent locally.
    You can type sentences like:
      - "Find the best math books"
      - "Search deeper for recent AI papers"
      - "Just links for FastAPI tutorials"
    """
    profile = default_profile or UserProfile(name="Ali", city="Lahore", topic="AI")
    while True:
        user_text = _cli_input_prompt()
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Simple heuristic to decide streaming vs sync: stream for longer queries
        # (You can adjust to always stream or base on config)
        do_stream = True

        # If you want to pass the last user message into personalised_prompt,
        # you would need to attach it to the context wrapper or SDK-specific field.
        # Some SDKs allow RunContextWrapper or context.extra; check yours.
        # For simplicity, we won't mutate the dataclass here.

        if do_stream:
            await run_streamed_query(profile, user_text)
        else:
            await run_sync_query(profile, user_text)

# ----------------------------
# Unit-test style helper (optional)
# ----------------------------
async def _self_test():
    """
    Quick self-test run to validate search tool and streaming.
    Useful for automated checks.
    """
    test_profile = UserProfile(name="TestUser", city="TestCity", topic="Testing")
    test_query = "best beginner python books 2024"
    logger.info("Running self-test...")
    # Try non-streamed run first (safer for automated tests)
    await run_sync_query(test_profile, test_query)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Web Search Agent (interactive)")
    parser.add_argument("--name", type=str, help="User name", default="Ali")
    parser.add_argument("--city", type=str, help="User city", default="Lahore")
    parser.add_argument("--topic", type=str, help="User topic of interest", default="AI")
    parser.add_argument("--non-interactive-query", type=str, help="Run a single query and exit")
    parser.add_argument("--self-test", action="store_true", help="Run a self-test and exit")
    args = parser.parse_args()

    user_profile = UserProfile(name=args.name, city=args.city, topic=args.topic)

    if args.self_test:
        asyncio.run(_self_test())
    elif args.non_interactive_query:
        # Run one query and print streaming result
        asyncio.run(run_streamed_query(user_profile, args.non_interactive_query))
    else:
        try:
            asyncio.run(interactive_loop(user_profile))
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
