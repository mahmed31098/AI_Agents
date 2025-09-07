# core.py
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunContextWrapper, AsyncOpenAI, OpenAIChatCompletionsModel


# Load env vars
_: bool = load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

# External client
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Models (shared across files)
light_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite", openai_client=external_client
)
main_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=external_client
)


@dataclass
class LocalContext:
    current_query: Optional[str] = None
    user_profile: Optional[dict] = None
    search_results: Optional[List[Dict]] = None
    citations: Optional[List[str]] = None

def dynamic_instructions(context: RunContextWrapper[LocalContext], agent: Agent) -> str:
    user_profile = context.context.user_profile or {
        "name": "User", "city": "L'Aquila", "interests": ["AI", "Research"]
    }
    dynamic_date = datetime.now().strftime("%Y-%m-%d")
    return f"""
    You are {agent.name}, a precise AI expert. Assist {user_profile['name']} from {user_profile['city']}.
    Follow the workflow strictly. Current Date: {dynamic_date}.
    """