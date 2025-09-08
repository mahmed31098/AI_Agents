import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings, function_tool

_: bool = load_dotenv(find_dotenv())  
#set_tracing_disabled(disabled=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

# first test
# @function_tool
# def greet (name:str) -> str:
#     """Greet a person by name."""
#     return f"Hello, {name}! How can I assist you today?"

# base_agent = Agent(
#     name="BaseAgent",
#     instructions="You are a helpful assistant.",
#     model=model,
#     tools=[greet]
# )

# res = Runner.run_sync(base_agent, "Hello, how are you?")



# second test

@function_tool
def get_weather (location:str) -> str:
    """Get the weather for a specific location."""
    return f"The weather in {location} is sunny."

base_agent = Agent(
    name="WeatherAgent",
    instructions="You are a helpful assistant.",
    model=model,
    tools=[get_weather]
)

res = Runner.run_sync(base_agent, "What's the weather like in L'Aquila?")

print(res)