# MAS-Start
Quick start examples for building multi-agent LLM systems using three frameworks.

---

## Setup

**1. Clone the repo**
```bash
git clone <repo-url>
cd MAS-Start
```

**2. Create a virtual environment (Python 3.12)**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install Chromium** (required for Magentic-One's web browsing agent)
```bash
playwright install --with-deps chromium
```

**5. Add your OpenAI API key**
```bash
cp .env.example .env
# then open .env and paste your key
```

---

## Run

```bash
# AutoGen AgentChat — fully connected, 3 agents
python AG.py

# LangGraph — star graph, 5 agents
python LC.py

# Magentic-One — pre-built autonomous team
python Magentic1.py
```

---

## What each file does

| File | Framework | Topology | Agents |
|---|---|---|---|
| `LC.py` | LangGraph | Star (hub + 4 spokes) | Hub, Researcher, Analyst, Critic, Writer |
| `AG.py` | AutoGen AgentChat | Fully connected | Researcher, Analyst, Summarizer |
| `Magentic1.py` | Magentic-One | Pre-built | Orchestrator, WebSurfer, FileSurfer, Coder, Terminal |

---

## References

- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [AutoGen documentation](https://microsoft.github.io/autogen/stable//index.html)
- [Magentic-One documentation](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/magentic-one.html)
