from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
import os
import asyncio
import requests
import re
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_: bool = load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Disable tracing for simplicity
set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@function_tool
def get_weather(location: str) -> str:
    print(f"\nFetching weather for: {location}\n")
    url = "https://api.tavily.com/search"

    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "query": f"current weather in {location}",
        "search_depth": "advanced",  # Use "advanced" to get richer answers
        "include_answer": True,
    }



    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result.get("answer", "No weather information found.")

        # Extract the Fahrenheit value (e.g., 80°F)
        match = re.search(r"(\d+)\s*°F", answer)
        if match:
            f_temp = int(match.group(1))
            c_temp = round((f_temp - 32) * 5 / 9)
            # Replace or append with Celsius
            return f"{answer} (approx. {c_temp}°C)"
        else:
            return answer
    except Exception as e:
        return f"Failed to get weather: {e}"

weather_agent = Agent(
    name="WeatherAgent",
    instructions="""You are a helpful assistant specialized in providing current weather information.""",
    model=llm_model,
    tools=[get_weather]
)

async def call_weather_agent():
    output = await Runner.run(
        starting_agent=weather_agent,
        input="Can you tell me the weather in L'Aquila today?"
    )
    print("\n Final Output:\n", output.final_output)

if __name__ == "__main__":
    asyncio.run(call_weather_agent())
