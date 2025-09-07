# report_writer.py
import os
from datetime import datetime
from agents import Agent
from core import main_model, light_model, LocalContext, dynamic_instructions


# Agent definition
report_writer_agent: Agent = Agent(
    name="Report Writer Agent",
    instructions=lambda ctx, agt: f"""
    {dynamic_instructions(ctx, agt)}

    Your role is to create the **final professional research report**.

    You will receive:
    1. A synthesized and citation-annotated draft response (with inline markers like [1], [2]).
    2. A validated APA-style reference list from the Source Checker Agent.

    Your tasks:
    - Transform the content into a **polished, professional research report**.
    - Organize the report into clear sections with headings, for example:
        - Title Page (with research title, date, author = user)
        - Executive Summary
        - Main Body (structured with sections/subsections from the synthesis)
        - Conclusion
        - References (APA, exactly as validated by Source Checker Agent)
    - Do not invent citations or references. Use only validated ones.
    - Ensure the report reads like an academic/professional document.

    Output the final report in plain text with clear section headings.
    """,
    model=main_model
)

# Helper to save report text
def save_report(report_text: str, folder: str = "reports") -> str:
    os.makedirs(folder, exist_ok=True)  # create folder if missing
    filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(folder, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return file_path
