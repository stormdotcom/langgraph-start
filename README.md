# LangGraph | OpenAI + Tools + SQLite Memory 

This project demonstrates a stateful LangGraph agent built with:

- LangChain OpenAI wrapper (ChatOpenAI)
- LangGraph 0.3.18 (graph orchestration)
- Tool calling (Search + Push Notification)
- SQLite checkpointing (persistent memory)
- CLI chat interface
- (Optional) Gradio UI

It recreates the “Day 3” agent pattern from the LangGraph course, but implemented entirely as a standalone Python script instead of Jupyter notebooks.

## 📁 File: openai-example.py

This script builds a tool-calling AI agent with persistent memory using LangGraph’s StateGraph.

## 🔧 1. Environment & Dependencies

The project uses these key libraries:

| Library | Purpose |
| ------- | ------- |
| langgraph | Graph-based agent orchestration |
| langchain-openai | OpenAI wrapper with tool calling |
| langgraph-checkpoint-sqlite | SQLite memory backend |
| langchain-community | Provides Serper search utility |
| requests | Required for the custom Pushover tool |
| python-dotenv | Load .env secrets |
| gradio | Optional: Web UI for interacting with agent |

We also constrain Python:

`requires-python = ">=3.10,<3.13"`

to avoid dependency resolution issues (OpenAI + LangChain markers differ for Python 3.13).

## 🔑 2. Environment Variables

The script loads secrets from .env:

```
OPENAI_API_KEY=sk-...
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
```

These values are required for:

- LLM calls
- Sending push notifications (optional tool)

## 🔍 3. Tools (Search + Push Notification)

The agent supports two tools, each defined via LangChain’s classic Tool class.

### 🔹 Search Tool (Google Serper)
```python
serper = GoogleSerperAPIWrapper()

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Search the Internet using Serper"
)
```

This tool lets the agent perform real-time search queries.

### 🔹 Push Notification Tool (Pushover)
```python
def push(text: str):
    requests.post(
        pushover_url,
        data={"token": pushover_token, "user": pushover_user, "message": text}
    )
    return "success"

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Send a notification using Pushover"
)
```

This allows the agent to send external alerts.

## 🧱 4. State Definition

The agent works over a shared state object shaped like:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`Annotated[..., add_messages]` makes LangGraph automatically merge LLM + tool responses into the message list.

## 🧠 5. LLM Node (Primary Chatbot Logic)
```python
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

This node:

- Reads the conversation history
- Calls OpenAI via ChatOpenAI
- Returns the assistant message
- LangGraph merges it into state

Tool calling is activated by binding tools:

```python
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

## 🔀 6. Graph Structure

The graph has:

- A chatbot node
- A ToolNode for executing tool calls
- Conditional routing based on whether the LLM requests a tool

### Conditional Routing
```python
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": END}
)
```

Meaning:

- If the LLM decides to use a tool → go to "tools" node
- If not → finish the graph (END)

After tool execution:
```python
graph_builder.add_edge("tools", "chatbot")
```

So it always returns to the LLM to continue reasoning.

## 🧩 7. SQLite Checkpoint Memory

Used to persist conversation state across multiple runs:

```python
conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)
```

Memory persists threads using:

```python
config={"configurable": {"thread_id": "day3-thread"}}
```

Each conversation thread gets its own saved state.

## 💬 8. CLI Chat Interface

You can run the agent purely in a terminal:

```python
def cli_chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )

        print("AI:", result["messages"][-1].content)
```

This is the simplest, most stable mode—no async, no UI layers.

## 🌐 9. Optional Gradio Web UI

The script includes an optional web UI:

```python
ui = gr.ChatInterface(
    fn=gradio_chat,
    title="LangGraph Day 3 Agent"
)
```

Launch with:

```python
if __name__ == "__main__":
    ui.launch()
