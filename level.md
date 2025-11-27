Below are **clean, professional, standalone Markdown files** for each learning level.
You can copy/paste each into separate `.md` files inside your project (or `/docs/` folder).

These are written like engineering documentation, not tutorial fluff.

---

# 📁 **LEVEL 1 — Enhancements (Beginner → Intermediate)**

Create file:

```
docs/LEVEL-1.md
```

---

# LEVEL 1 — Enhancing the Basic LangGraph Agent

Beginner → Intermediate Improvements

This document describes small but important upgrades you can add to your existing LangGraph Day-3 agent to understand **graph behavior, state merging, and debugging**.

---

# ✅ 1. Add a State Logger Node

A logger node is the simplest way to understand how LangGraph updates state between nodes.

### ➤ Code

```python
def logger(state: State):
    print("\n[STATE UPDATE]")
    print(state)
    return state
```

Add node + edge:

```python
graph_builder.add_node("logger", logger)
graph_builder.add_edge("chatbot", "logger")
graph_builder.add_edge("logger", END)
```

### ➤ What you learn

* How state propagates through the graph
* How messages get appended via `add_messages`
* How tool results appear in state

---

# ✅ 2. Add a Memory Summary Tool

The agent generates a running summary of the conversation, which is useful for:

* long-term memory
* compressing chat history
* context limits

### ➤ Code

```python
summary_llm = ChatOpenAI(model="gpt-4o-mini")

def summarize_messages(text: str):
    result = summary_llm.invoke([
        {"role": "system", "content": "Summarize this text."},
        {"role": "user", "content": text}
    ])
    return result.content

tool_summary = Tool(
    name="conversation_summary",
    func=summarize_messages,
    description="Summarize the entire conversation"
)
```

Add to tools list:

```python
tools.append(tool_summary)
```

### ➤ What you learn

* LLM-as-tool pattern
* How tools can create new state
* How agents can compress memory

---

# ✅ 3. Add a Note-Taking Tool

Let the user ask:

> "save this as a note"

### ➤ Code

```python
def save_note(text: str):
    conn.execute("INSERT INTO notes (text) VALUES (?)", (text,))
    conn.commit()
    return "Note added."

tool_save_note = Tool(
    name="save_note",
    func=save_note,
    description="Save text to persistent notes"
)

tools.append(tool_save_note)
```

Create table:

```sql
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT
);
```

### ➤ What you learn

* Adding domain-specific memory
* Using database tools
* Building real app functionality

---

# 📚 Summary

Level 1 teaches:

| Concept      | Skill               |
| ------------ | ------------------- |
| Logging node | State introspection |
| Summary tool | Memory compression  |
| Note tool    | Tool → DB workflows |

These are foundational LangGraph patterns before moving on to multi-agent designs.

---

---

# 📁 **LEVEL 2 — Multi-Node Graphs (Intermediate)**

Create:

```
docs/LEVEL-2.md
```

---

# LEVEL 2 — Adding Multi-Node Graph Logic

Intermediate Graph Design

These upgrades move beyond simple chatbot → tool flows and introduce **graph routing, preprocessing, and multiple nodes**, which is the core of LangGraph design.

---

# ✅ 1. Add a Guardrail (Input Filter) Node

Intercept messages before they reach the LLM.

### ➤ Code

```python
def guard(state: State):
    msg = state["messages"][-1]["content"]

    if "badword" in msg.lower():
        return {"messages": [
            {"role": "assistant", "content": "Please avoid inappropriate language."}
        ]}
    return state
```

Add routing:

```python
graph_builder.add_node("guard", guard)
graph_builder.add_edge(START, "guard")
graph_builder.add_edge("guard", "chatbot")
```

### ➤ What you learn

* Pre-processing steps
* Input validation
* Branching and node chaining

---

# ✅ 2. Add a Retrieval (RAG) Tool

This integrates vector search (FAISS, Chroma, etc.)

### ➤ Code

```python
def retrieve(query: str):
    results = vector_db.similarity_search(query)
    return str(results)

tool_retrieve = Tool(
    name="retrieve_docs",
    func=retrieve,
    description="Search internal knowledgebase"
)
```

### ➤ What you learn

* Hybrid LLM-RAG systems
* Using vector databases
* Building knowledge assistants

---

# 📚 Summary

Level 2 teaches:

| Concept               | Skill                 |
| --------------------- | --------------------- |
| Guardrail             | Graph entry filtering |
| RAG tool              | Knowledge retrieval   |
| Multi-node sequencing | Real agent flows      |

---

---

# 📁 **LEVEL 3 — Advanced Agent Patterns**

Create:

```
docs/LEVEL-3.md
```

---

# LEVEL 3 — Multi-Agent Architectures

Intermediate → Advanced

These upgrades introduce **agent decomposition**, the core pattern behind:

* OpenAI’s Agents
* AutoGPT
* BabyAGI
* LangGraph’s worker-evaluator

---

# ✅ 1. Add a Planner Node

Planner decides what to do.

### ➤ Code

```python
def planner(state: State):
    plan = llm.invoke([
        {"role": "system", "content": "You are a planner."},
        {"role": "user", "content": state["messages"][-1]["content"]}
    ])
    return {"plan": plan.content}
```

---

# ✅ 2. Add a Worker Node

Worker executes the plan with tools.

### ➤ Concept

Planner → Worker → Tools → Worker → Planner → END

This teaches how loops work in LangGraph.

---

# 📚 Summary

Level 3 teaches:

| Concept              | Skill                      |
| -------------------- | -------------------------- |
| Planner → Worker     | Multi-agent design         |
| Cyclic edges         | Loops in LangGraph         |
| Separation of duties | Robust agent orchestration |

---

---

# 📁 **LEVEL 4 — Real-World Tools & Automation**

Create:

```
docs/LEVEL-4.md
```

---

# LEVEL 4 — Browser Automation & File Tools

Advanced Tools

These upgrades integrate the agent with **real-world data and actions**.

---

# ✅ 1. Playwright Web Automation Tool

Let the agent visit websites, click buttons, scrape data.

### ➤ Code

```python
from playwright.sync_api import sync_playwright

def browse(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        return page.content()
```

### ➤ What you learn

* Browser automation
* Agent-driven web actions
* Handling long-running tools

---

# ✅ 2. File System Tools

```python
def write_file(data: dict):
    with open(data["path"], "w") as f:
        f.write(data["content"])
    return "file saved"
```

### ➤ What you learn

* Safe real-world actions
* External side effects in graphs

---

# 📚 Summary

Level 4 teaches:

| Concept         | Skill                     |
| --------------- | ------------------------- |
| Playwright tool | Browser automation        |
| File tools      | Local action execution    |
| Side effects    | Real-world agent behavior |

---

---

# 📁 **LEVEL 5 — Deploying a Real AI System**

Create:

```
docs/LEVEL-5.md
```

---

# LEVEL 5 — Turning It Into a Real Application

Expert Level

---

# ✅ 1. Add FastAPI Backend

Expose your agent as:

* `/chat`
* `/invoke`
* `/tools`
* `/state`
* `/history`

---

# ✅ 2. React Frontend

Create a chat UI that communicates with FastAPI.

---

# ✅ 3. Add LangSmith Tracing

Full graph visualization and tracing:

```
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
```

---

# 📚 Summary

Level 5 teaches:

| Concept              | Skill                     |
| -------------------- | ------------------------- |
| API deployment       | Serve LangGraph           |
| Frontend integration | Build full-stack AI apps  |
| Observability        | Debug, monitor, visualize |

---

