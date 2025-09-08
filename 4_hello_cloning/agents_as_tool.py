import os
import asyncio

from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, RunContextWrapper
from datetime import datetime

_:bool = load_dotenv(find_dotenv()) 

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "") # for tracing
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
external_client: AsyncOpenAI = AsyncOpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

light_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-flash-lite", openai_client=external_client)
main_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)
pro_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gemini-2.5-pro", openai_client=external_client)

planning_agent: Agent = Agent( name = "Planning Agent", instructions = "You are a planning assistant, Look at user request and use scientific reasoning to create a plan of action.", model = pro_model )
web_search_agent: Agent = Agent( name = "Web Search Agent", instructions = "You are a web search assistant. Look at user request, plan and use web search to find relevant information.", model = main_model )
reflective_agent: Agent = Agent( name = "Reflective Agent", instructions = "You are a reflective assistant. Look at user request and reflect on the best approach to take.", model = light_model )

@function_tool
async def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def dynamic_instructions(context: RunContextWrapper, agent: Agent) -> str:
    dynamic_date=datetime.now().strftime("%Y-%m-%d")
    prompt= f"""
    You are an orchestrator agent. You manage specialized agents to help users. Your main goal is deep search for each user query. We follow a structured process for each deep search request.
    1. do planning
    2. spawn multiple web search agents
    3. perform reflection to decide if deep goal is achieved
    Finally get reflection to know if the task is achieved. Current Date {dynamic_date}.
    """
    return prompt

# this way our design pattern will improve, 
orchestrator_agent: Agent = Agent( name = "Orchestrator Agent", instructions = dynamic_instructions, model = main_model,
                                  tools=[
                                      planning_agent.as_tool(
                                          tool_name="planning_agent",
                                          tool_description="A planning agent that uses scientific reasoning to plan next steps."),
                                      web_search_agent.as_tool(
                                          tool_name="web_search_agent",
                                          tool_description="A web search agent that uses web search to find relevant information."),
                                      reflective_agent.as_tool(
                                          tool_name="reflective_agent",
                                          tool_description="A reflective agent that reflects on the best approach to take.")
                                  ] )


async def call_planning_agent():
    res = await Runner.run(orchestrator_agent, "Do deep search for Lead Generation Systems for a Tax Company in US")
    print(res.final_output)

# Now we will see the trace in logs in OpenAI dashboard

if __name__ == "__main__":
    asyncio.run(call_planning_agent())