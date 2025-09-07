from agents import Agent
from core import main_model, light_model, LocalContext, dynamic_instructions
from synthesis_agent import synthesis_agent
from report_writer import report_writer_agent

# Web Search Agent
web_search_agent: Agent = Agent(
    name="Web Search Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}
    Execute web searches for assigned sub-tasks.
    Summarize: Key findings + sources.
    Store results in context for citations.
    """,
    model=main_model
)

# Reflective Agent
reflective_agent: Agent = Agent(
    name="Reflective Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}
    Review all gathered results and identify themes, patterns, contradictions, and gaps before passing them to the Synthesis Agent.
    """,
    model=light_model
)

# Citations Agent
citations_agent: Agent = Agent(
    name="Citations Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}
    From the search results in context, generate numbered in-text citation markers like [1], [2], etc.
    Ensure that every factual claim in the synthesized response has a numbered citation next to it.
    At the end, generate a reference list where each entry matches its number, formatted in APA style.
    Example format:
    In-text: "Climate change accelerates ice melting [1]."
    References:
    [1] Smith, J. (2022). *Understanding Climate Change*. Climate Press. https://example.com
    """,
    model=light_model
)


#    And finally generate APA style citations for all credible numbered citations along with their numbers
#   If you find conflicting information, highlight it clearly: for example, "Source A says X, but Source B says Y" and let users know there's disagreement in the context.
    


source_checker_agent: Agent = Agent(
    name="Source Checker Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}
    Review all sources for credibility and relevance. 
    Rate sources as High (.edu, .gov, major news), Medium (Wikipedia, industry sites), or Low (blogs, forums) and warn users about questionable information. 
    For all credible sources, generate properly formatted APA citations.
    Ensure each citation is numbered [1], [2], etc., to match the in-text references from the Synthesis Agent.
    Use the following APA format:
    [1] AuthorLastName, F. (Year). *Title of the source*. Publisher or Website. URL
    Clearly exclude or flag low-credibility sources, and do not include them in the final reference list.
    """,
    model=light_model
)


# Orchestrator Agent coordinates all other agents
orchestrator_agent: Agent = Agent(
    name="Orchestrator Agent",
    instructions=lambda ctx, agt: dynamic_instructions(ctx, agt) + """
    You are the lead. Receive plan from Planning Agent.
    Assign sub-tasks (maximum 2) to Web Search Agent (call as tool).
    Only after all web searches, call Reflective Agent (call as tool),
    After Reflection, call Synthesis Agent (call as tool),
    After Synthesis Agent, call Citations Agent (call as tool) to insert inline citation numbers and prepare the reference list.
    Then call the Source Checker Agent (call as tool) to validate sources and regenerate APA-formatted references.
    After the Source Checker Agent validates sources and everything is complete and verified, compile the final response.
    Call the Report Writer Agent (call as tool) to generate the final polished research report.
    """,
    model=main_model,
    tools=[web_search_agent.as_tool(tool_name="web_search_agent", tool_description="A web search agent that uses web search to find relevant information."),
           reflective_agent.as_tool(tool_name="reflective_agent", tool_description="A reflective agent that reflects on the best approach to take."),
           synthesis_agent.as_tool(tool_name="synthesis_agent", tool_description="A synthesis agent that combines insights from all agents to create a cohesive response."),
           citations_agent.as_tool(tool_name="citations_agent", tool_description="A citations agent that generates citations from search results."),
           source_checker_agent.as_tool(tool_name="source_checker_agent", tool_description="A source checker agent that checks the credibility of sources and generates APA style citations."),
           report_writer_agent.as_tool(tool_name="report_writer_agent", tool_description="Finalizes the professional research report with APA references.")]
)
           