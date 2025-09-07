from agents import Agent
from core import main_model, light_model, LocalContext, dynamic_instructions


synthesis_agent: Agent = Agent(
    name="Synthesis Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}

    Look at the response from Reflective Agent to create a cohesive response.
    This involves synthesizing all research findings and organizing them into clear sections with themes, trends, and key insights rather than just listing facts.

    While writing the synthesized response, ensure that every factual claim is followed by an inline citation like [1], [2], etc., corresponding to sources provided by the Citations Agent.
    Do not invent citation numbers. Use only numbers for which a source is available.
    Ensure the text reads naturally while integrating citations.

    """,
    model=main_model
)