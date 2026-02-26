"""
AutoGen AgentChat Multi-Agent Example — Fully Connected (3 agents)
──────────────────────────────────────────────────────────────────────────────
Fully connected topology: every agent sees every message (shared group chat).
Agents take turns via RoundRobinGroupChat.

    Researcher ◄──► Analyst ◄──► Summarizer
         ▲                            │
         └────────────────────────────┘
              (all messages shared)

Flow:  Researcher → Analyst → Summarizer → TERMINATE

Install:
    pip install "autogen-agentchat" "autogen-ext[openai]" python-dotenv
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# ── Model client ──────────────────────────────────────────────────────────────
client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

# ── Agents ────────────────────────────────────────────────────────────────────
researcher = AssistantAgent(
    name="Researcher",
    model_client=client,
    system_message="You are a researcher. Answer the question with relevant facts.",
    # tools=[my_tool]  ← add this to give the agent tools; it decides when to call them
)

analyst = AssistantAgent(
    name="Analyst",
    model_client=client,
    system_message="You are an analyst. Identify key patterns and implications from the research.",
    # tools=[my_tool]  ← each agent can have its own tools, or share the same ones
)

summarizer = AssistantAgent(
    name="Summarizer",
    model_client=client,
    # tools=[my_tool]  ← tool calls are handled automatically; no graph wiring needed
    system_message=(
        "You are a summarizer. Condense the discussion into 3 clear bullet points. "
        "End your message with TERMINATE."
    ),
)

# ── Team ──────────────────────────────────────────────────────────────────────
team = RoundRobinGroupChat(
    participants=[researcher, analyst, summarizer],
    termination_condition=TextMentionTermination("TERMINATE"),
)


# ── Run ───────────────────────────────────────────────────────────────────────
async def main():
    task = "What are the main benefits and challenges of adopting renewable energy?"
    print(f"Task: {task}\n{'─' * 60}\n")
    await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
