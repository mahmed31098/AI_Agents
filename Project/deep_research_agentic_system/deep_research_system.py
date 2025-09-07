import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from core import main_model, light_model, LocalContext, dynamic_instructions
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from report_writer import save_report
from research_agents import orchestrator_agent, web_search_agent, reflective_agent, citations_agent
from planning_agent import planning_agent
from report_writer import save_report

requirement_gathering_agent: Agent = Agent(
    name="Requirement Gathering Agent",
    instructions=lambda ctx, agt: dynamic_instructions(ctx, agt) + """
    You are the first agent. Interact with the user to clarify the research query.
    Ask for details if needed (e.g., scope, depth) ONLY ONCE and then use the `handoff_to_planning_agent` tool to transfer the query to the Planning Agent.
    Do not perform searches yourself.
    """,
    model=light_model,
    handoffs=[handoff(planning_agent)] 
)


# Main function to initiate the flow
async def main():
    initial_query = input("Enter your research query: ")
    local_context = LocalContext(current_query=initial_query)
    current_query = initial_query

    # Turn 1: Requirement Gathering with user interaction
    print("\n[Turn 1: Requirement Gathering Agent]")
    r1 = await Runner.run(requirement_gathering_agent, current_query, context=local_context)
    print(r1.final_output)
        
    # Simulate interaction: Prompt user for confirmation or more details
    user_response = input("\nYour response: ")

    t2_input = r1.to_input_list() + [{"role": "user", "content": user_response }]

    local_context.current_query = current_query

    r2 = await Runner.run(r1.last_agent, t2_input, context=local_context)

    final_text = r2.final_output
    print("\n\nFinal Report:\n", final_text)

    saved_path = save_report(final_text)
    print(f"\n✅ Report saved at: {saved_path}")

    #print( "\n\n\nFinal Response from from Orchestrator:\n\n\n", r2.final_output)

if __name__ == "__main__":
    asyncio.run(main())
