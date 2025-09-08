import os
import asyncio
from typing import Callable
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from openai.types.responses import ResponseTextDeltaEvent # downloaded for step 9

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

# to understand how dynamic instructions work
@function_tool
async def search(local_context: RunContextWrapper[UserContext], query: str) -> str: #using dynamic typing I can pass my object inside wrapper


    import time
    time.sleep(30)  # Simulating a delay for the search operation

    # print("\n\n SOME DATA\n\n", local_context.context.email, "\n\n")  #he inside local_context -> context -> there is username and email both can be accessed
    # here as object as being given to the wrapper now we can access whats inside the context like email and username
    #Db or API call can be made here to fetch data based on the query -> username is available in local_context
    return "No results found." # IF WE DON'T want to output any search results


# 1. first thing we understood in dynamical programming

# math_agent = Agent(
#     name="Genius",
#     instructions="You are a math expert.", # this is part of out system prompt
# # when list goes into the llm all te data is passed, for example we have a list which contains the dictionary of messages
# # [{"role": "system", "content": "What are a math expert?"}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], the assistant is the response from llm and its content is what we get as the final output
# # llm will consider system instructions as the most important because rest of the chat is between user and assistant
# # so its for wish if you want to pass an instruction or a callable,
# # with callable your instructions will be generated in real-time based on the context of the conversation
#     model=ll_model, tools=[search])

# # 2. next we make a callable function

# def get_user_special_prompt() -> str: # we have just single requirement in the output that it must return the string
#     # who is th user?
#     # who is the agent?
#     return "You are a math expert."

# # 3. adjustments

# math_agent = Agent(
#     name="Genius",
#     instructions= get_user_special_prompt, # we just need to pass the callable function we arent writing get_user_special_prompt() here to explicitly call it.
#     # when we will run the Runner it will call this function by itself
#     model=llm_model, tools=[search])

# # 4. final adjustments

# # the following is our callabale, here we are passing local context and agent class
# def get_user_special_prompt(local_context, agent) -> str: # hirarchy defined that local context is always first argument, either name it local_context or something else like special_context or context,
#     # and what ever send in form of return will be passed as system prompt to the llm
#     # who is the user?
#     # who is the agent?
#     print(f"\nUser: {local_context},\n Agent: {agent}\n")  # This will print the user context and agent name
#     return "You are a math expert."

# math_agent = Agent(
#     name="Genius",
#     instructions= get_user_special_prompt, # we just need to pass the callable function we arent writing get_user_special_prompt() here to explicitly call it.
#     # when we will run the Runner it will call this function by itself
#     model=llm_model, tools=[search])



# 5. adjustments

# we made the function asynchronous and added type hinting to local_context
# this function get_user_special_prompt is called Callable which takes three inputs RunContextWrapper[TContext], agent[TContext] and MaybeAwaitable[str] (MaybeAwaitable is used to indicate that the function can return a string or an awaitable object that resolves to a string)
async def get_user_special_prompt(local_context: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str: # we define type of local_context as RunContextWrapper[UserContext] to access the context attributes like email and username
# we can either just write agent there or Agent[UserContext] to define the type of agent, here we are using UserContext as the type of context for the agent
    print(f"\nUser: {local_context.context},\n Agent: {agent.name}\n")  # This will print the user context and agent name
    return f"You are a math expert. User: {local_context.context.username}, Agent: {agent.name}, Please assist me with my math problem."  # This will return a string that includes the username and email if available
# by doing the above I am telling my agent that your username is abdullah and your agent name is Genius
#because in the below the name and instruction arent being passed explicitly, they are being passed in math_agent: Agent class

math_agent = Agent(
    name="Genius",
    instructions= get_user_special_prompt, # we just need to pass the callable function we arent writing get_user_special_prompt() here to explicitly call it.
    # when we will run the Runner it will call this function by itself
    model=llm_model, tools=[search])    


# async def call_agent(): # part of 5

#     user_context = UserContext(username="abdullah")  # Create an instance of UserContext

#     output = await Runner.run(
#         starting_agent=math_agent,
#         #input="Hi", # 6. run with this first
#         #input = "what is your name ? and who is the user?", # output "My name is Genius and the user's name is Abdullah."
#         input= "search for the best maths books", #our search agent takes alot of timeand user needs to wait for the response too long, if it would have been deep search agent it would have taken even more time.
        
#         context=user_context  # Pass the user context to the agent
#     )
#     print(output.final_output)



# for 1 till 4
# async def call_agent():

#     user_context = UserContext(username="abdullah")  # Create an instance of UserContext


#     output = await Runner.run(
#         starting_agent=math_agent,
#         input="What is latest llm model from China? Can you search the web for it?", #QS OF THE USER
#         #context = {"username": "abdullah"} # did without dataclass (for dataclass we created a class above and made an instance of it here in this function)
#         context=user_context  # Pass the user context to the agent
#     )
#     print(output.final_output)


# 8 Just to start the sreaming
async def call_agent():

    user_context = UserContext(username="abdullah")  # Create an instance of UserContext

    output = Runner.run_streamed(
        starting_agent=math_agent,
        input= "search for the best maths books", #our search agent takes alot of timeand user needs to wait for the response too long, if it would have been deep search agent it would have taken even more time.
        context=user_context  # Pass the user context to the agent
    )
    async for output in output.stream_events():

        # step 8.5 will show the deatiled output
         #print(output,"\n\n\n")



        # step 9, will show the raw response event 
        if output.type == "raw_response_event" and isinstance(output.data, ResponseTextDeltaEvent):
            print(output.data.delta, end="", flush=True)

       


asyncio.run(call_agent())









# # we have a variable and it stype is str
# user: str = "abdulah"

# # this function has type callable because its a typing function
# prompt_function: Callable[[str], str] # this function takes as input a string and provides output as string

# def get_prompt_function(user_input: str) -> str:
#     return f"Hello, {user_input}!"