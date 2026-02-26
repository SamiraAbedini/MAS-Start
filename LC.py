"""
LangGraph Multi-Agent Star Graph (5 agents)
──────────────────────────────────────────────────────────────────────────────
Star topology: one central Hub connected to 4 specialist spokes.

                         START
                           │
                         [Hub]  ◄──────────────────────┐
                        ╱  │  ╲  ╲                     │
                       ▼   ▼   ▼   ▼                   │
                  Researcher Analyst Critic Writer     │
                       │   │   │   │                   │
                       └───┴───┴───┘ ──────────────────┘
                                          (each spoke returns to Hub)
                         [Hub]
                           │
                          END

Flow:
  START → Hub → Researcher → Hub → Analyst → Hub → Critic → Hub → Writer → Hub → END

The Hub keeps track of which spokes have been visited. Once all four have
contributed, it synthesizes their outputs into a final answer.

Install:
    pip install langgraph langchain-openai python-dotenv
──────────────────────────────────────────────────────────────────────────────
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
# To add tools: llm = llm.bind_tools([my_tool])  ← the LLM will emit tool_calls in its response


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # each node appends its message
    next: str                                 # hub writes this to pick the next spoke
    visited: list[str]                        # spokes that have already responded


# ── Specialist definitions (spoke name → system prompt role) ──────────────────
SPECIALISTS = {
    "researcher": (
        "You are a knowledgeable researcher. "
        "Provide relevant facts, data, and background on the topic."
    ),
    "analyst": (
        "You are a data analyst. "
        "Examine the information gathered so far and identify key patterns, "
        "trends, and deeper implications."
    ),
    "critic": (
        "You are a critical thinker. "
        "Identify weaknesses, counterarguments, risks, or missing perspectives "
        "in the discussion so far."
    ),
    "writer": (
        "You are a skilled writer. "
        "Synthesize all previous contributions into a clear, concise, "
        "well-structured response."
    ),
}


# ── Hub node (center of the star) ─────────────────────────────────────────────
def hub(state: AgentState) -> dict:
    """
    Checks which spokes have been visited.
    - If a spoke is still pending  → route to it.
    - If all spokes are done       → produce the final synthesis and route to END.
    """
    visited = state.get("visited", [])

    for name in SPECIALISTS:
        if name not in visited:
            print(f"\n[HUB] routing → {name.upper()}")
            return {"next": name}

    # All spokes have contributed — synthesize
    print("\n[HUB] all specialists done → synthesizing final answer")
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an orchestrator. All specialist agents have shared "
                    "their perspectives. Synthesize everything into one final, "
                    "well-structured answer."
                ),
            }
        ]
        + state["messages"]
    )
    return {"messages": [response], "next": END}


# ── Spoke factory ─────────────────────────────────────────────────────────────
def make_spoke(name: str, system_prompt: str):
    """Returns a node function for one specialist spoke."""

    def spoke(state: AgentState) -> dict:
        print(f"\n[{name.upper()}] working...")
        response = llm.invoke(
            [{"role": "system", "content": system_prompt}] + state["messages"]
        )  # If the LLM emits a tool_call, a ToolNode in the graph will execute it
        return {
            "messages": [response],
            "visited": state.get("visited", []) + [name],
        }

    spoke.__name__ = name
    return spoke


# ── Routing function (reads state["next"]) ────────────────────────────────────
def route_from_hub(state: AgentState) -> str:
    return state["next"]


# ── Build the star graph ───────────────────────────────────────────────────────
graph = StateGraph(AgentState)

# Center node
graph.add_node("hub", hub)

# Spoke nodes — each edge returns to the hub (the star's return paths)
# To add tools: graph.add_node("tools", ToolNode([my_tool]))  then add a conditional edge: spoke → tools → spoke
for name, prompt in SPECIALISTS.items():
    graph.add_node(name, make_spoke(name, prompt))
    graph.add_edge(name, "hub")  # ← spoke always feeds back to center

# Entry point
graph.add_edge(START, "hub")

# Hub uses conditional edges to dispatch to any spoke or END
graph.add_conditional_edges(
    "hub",
    route_from_hub,
    {**{name: name for name in SPECIALISTS}, END: END},
)

app = graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task = "What are the main benefits and challenges of adopting renewable energy?"
    print(f"Task: {task}\n{'─' * 60}")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "visited": [],
            "next": "",
        }
    )

    print(f"\n{'═' * 60}")
    print("FINAL SYNTHESIS")
    print(f"{'═' * 60}")
    print(result["messages"][-1].content)
