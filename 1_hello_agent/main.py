
# AGENT LEVEL MODEL CONFIGURATION
# This code demonstrates how to configure an AI agent with a specific LLM model using the Agentic AI framework. 


from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool #(its a method for tool calling)
import os
import asyncio

from dotenv import load_dotenv, find_dotenv
_: bool = load_dotenv(find_dotenv())  # Load environment variables from .env file

gemini_api_key: str | None = os.environ.get("GOOGLE_API_KEY")

set_tracing_disabled(True)


# 1. Which LLM Service? 
external_client: AsyncOpenAI = AsyncOpenAI( 
    #we will use this AsyncOpenAI class telling which provider am I using
    api_key=gemini_api_key, # Api Key and route through baseurl is provided which tells with this key and following this route you can do XYZ
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # google has provided this url so that with the key and this path it can be used
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", #https://aistudio.google.com/app/prompts/new_chat you can see models of gemini from here
    openai_client=external_client # both client and model name provided
)

# 2.5. Function Tool (optional)
@function_tool # function_tool decorator is used to define a function that can be called by the agent
def add_numbers(a: int, b: int) -> int:
    print(f"\n\nAdding numbers: {a} + {b}\n\n")
    return a + b

@function_tool
def subtract_numbers(a: int, b: int) -> int:
    print(f"\n\nSubtracting numbers: {a} - {b}\n\n")
    return a - b

@function_tool
def some_cool_function(a: int, b: int) -> int:
    print(f"\n\nMultiplying numbers: {a} * {b}\n\n")
    return a * b


# 3. Create an Agent (Running synchronously)

math_agent: Agent = Agent(name="MathAgent", # any agent can be defined
                     instructions="""You are a helpful math assistant.  
                     Always answer in Italian""", # (system prompt) here we do either prompt engineering or context engineering
                     model=llm_model, 
# Before providing the final output, the LLM will first attempt to call any available tools (functions).
# If a tool is called, its result will be used, and then the LLM will generate the final answer for the user.


                     tools=[add_numbers, subtract_numbers, some_cool_function] # tools are functions that can be called by the agent
# The agent is responsible for calling the tools, not the runner. The llm_model is used by the agent to generate responses.

                        )

# the agent name and instructions donot go to the LLM, they are just for our reference
# name and model are required in Agent class, instructions are optional
# can I define llm_model at runtime during Async runner.run? Yes it can be set at three different levels , Agent level, Run Level and Global Level
# availble at learn-agentic-ai/01_ai_agents_first/05_model_configuration on my forked github repo


async def call_agent():
    # 4. Create a Runner (to run the agent asynchronously)
    #runner.run() is an async function that runs the agent
    #await is used ky jo background us ko chlany dy and only result return karay
    output = await Runner.run(starting_agent=math_agent, # agent is passed to the runner
                                     input="What is 2 + 2 and what is 7 - 9 and what is 9 * 7?" # input is passed to the agent
#although we have the some_cool_function defined, we are not using it in the input, but it is available to the agent
# we can use the tools in the input, but we are not using it here, we are just testing the agent's ability to call the tools
# by running the folllowing above file it runs the add_numbers and subtract_numbers functions but not the some_cool_function and responded as
# Non posso rispondere a 9 * 7 in quanto non ho gli strumenti adatti.
# which means I cannot answer 9 * 7 as I do not have the right tools                                   
                                    )
    print(output.final_output)

# await call_agent()  # we cant use await outside an async function, so we define an async function and call it

asyncio.run(call_agent()) # run the async function

# # 4. Create a Runner (to run the agent synchronously)
# output = Runner.run_sync(starting_agent=math_agent, # agent is passed to the runner
#                          input="What is 2 + 2?" # input is passed to the agent
#                             )

# print(output.final_output) 

# def main():
#     print("Hello from hello-agent!")


# if __name__ == "__main__":
#     main()
