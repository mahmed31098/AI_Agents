import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper #context class we imported

_: bool = load_dotenv(find_dotenv())

google_api_key: str | None = os.environ.get("GOOGLE_API_KEY")

# Tracing disabled
set_tracing_disabled(disabled=True)

# 1. Which LLM Service?
external_client = AsyncOpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# 2. Which LLM Model?
llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@dataclass
class UserContext:
    username: str
    email: str | None = None

# 1. just to understand how context works
# @function_tool
# async def search(local_context: RunContextWrapper, query: str) -> str: # runcontextwrapper is always first argument

#     # Here we can access the context of the agent (This is how context is passed to the function)
#     print("\n\n SOME DATA\n\n", local_context, "\n\n") # by defining in Run
#     # here in the output you will get the ToolContext which contains alot of other data as well
#     #Db or API call can be made here to fetch data based on the query -> username is available in local_context
#     return "No results found." # IF WE DON'T want to output any search results

    # Here you would implement the actual search logic, e.g., using an API to search the web.
    #print(f"Searching for: {query}")
    #return f"Search result for: {query}"  # Placeholder response


# 2. to understand how dynamic instructions work
# @function_tool
# async def search(local_context: RunContextWrapper, query: str) -> str: #using dynamic typing I can pass my object inside wrapper

#     print("\n\n SOME DATA\n\n", local_context.context, "\n\n")  #he inside local_context -> context -> there is username and email both can be accessed
#     # here I will get the UserContext whole object as an output including its attributes
#     #Db or API call can be made here to fetch data based on the query -> username is available in local_context
#     return "No results found." # IF WE DON'T want to output any search results



# 3. to understand how dynamic instructions work
@function_tool
async def search(local_context: RunContextWrapper[UserContext], query: str) -> str: #using dynamic typing I can pass my object inside wrapper

    print("\n\n SOME DATA\n\n", local_context.context.email, "\n\n")  #he inside local_context -> context -> there is username and email both can be accessed
    # here as object as being given to the wrapper now we can access whats inside the context like email and username
    #Db or API call can be made here to fetch data based on the query -> username is available in local_context
    return "No results found." # IF WE DON'T want to output any search results

math_agent = Agent(name="Genius", model=llm_model, tools=[search])

async def call_agent():

    user_context = UserContext(username="abdullah")  # Create an instance of UserContext


    output = await Runner.run(
        starting_agent=math_agent,
        input="What is latest llm model from China? Can you search the web for it?", #QS OF THE USER
        #context = {"username": "abdullah"} # did without dataclass (for dataclass we created a class above and made an instance of it here in this function)
        context=user_context  # Pass the user context to the agent
    )
    print(output.final_output)

asyncio.run(call_agent())
