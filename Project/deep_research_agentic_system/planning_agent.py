from agents import Agent, handoff
from core import main_model, light_model, LocalContext, dynamic_instructions
from research_agents import orchestrator_agent  # only if orchestrator is in research_agents

planning_agent: Agent = Agent(
    name="Planning Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}
    Break the clarified query into max 2 specific search tasks and then use the `handoff_to_orchestrator_agent` tool to transfer the query to the Orchestrator Agent.
    """,
    model=light_model,
    handoffs=[handoff(orchestrator_agent)]
)
