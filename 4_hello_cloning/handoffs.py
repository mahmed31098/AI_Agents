import os

from agents_as_tool import dynamic_instructions
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff

_:bool = load_dotenv(find_dotenv()) 

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "") # for tracing
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
external_client: AsyncOpenAI = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

main_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)
pro_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=external_client)

planning_agent: Agent = Agent( name = "Planning Agent", 
                        instructions = """
                        You are a planning assistant.
                        Look at the user request and use scientific reasoning to create a plan of action. 
                        """,
                        model = main_model ) # i should have given the pro_model but it wasnt working so I used the main and it worked

orchestrator_agent: Agent = Agent( name = "Deep Agent", 
                                  instructions = "You are a helpful assistant which can answer the questions but for planning, delegates to Planning Agent.",
                                
                                  model = main_model,
                                  #tools=[],
                                  handoffs=[planning_agent]
)

res = Runner.run_sync(orchestrator_agent, "Plan for Lead Generation Systems for a Tax Company in US") # for planning we can see the last agent is planning agent while for things without planning like "hi" our main agent which is the orchestrator agent responds 
print("\nLast Agent:", res.last_agent.name)
print("\n\nFinal Output:", res.final_output)


# uv run handoffs.py