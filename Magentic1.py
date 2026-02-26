"""
Magentic-One Multi-Agent Example
──────────────────────────────────────────────────────────────────────────────
Magentic-One is a pre-built multi-agent team from Microsoft that includes:
  • Orchestrator   – plans the task and delegates to specialists
  • WebSurfer      – browses the web
  • FileSurfer     – reads local files
  • Coder          – writes and executes code
  • ComputerTerminal – runs shell commands

The MagenticOne helper class wires all agents together automatically.
You simply give it a task and it figures out which agents to use.

⚠️  Safety note: Magentic-One can execute code and browse the internet.
    Run it inside a container or VM when using the Coder/Terminal agents.

Install:
    pip install "autogen-agentchat" "autogen-ext[magentic-one,openai]" python-dotenv
    playwright install --with-deps chromium   # only needed for WebSurfer
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console

load_dotenv()

# ── Model client ──────────────────────────────────────────────────────────────
client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
)


# ── Run ───────────────────────────────────────────────────────────────────────
async def main():
    # MagenticOne creates and connects all specialist agents automatically.
    m1 = MagenticOne(client=client)

    task = "What are the main benefits of multi-agent AI systems? Write a short summary."
    print(f"Task: {task}\n{'─' * 60}\n")

    # run_stream yields each agent message as it is produced.
    result = await Console(m1.run_stream(task=task))
    print(f"\n{'─' * 60}\nFinal answer:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
