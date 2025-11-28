import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

from langgraph.checkpoint.sqlite import SqliteSaver
import gradio as gr
import os
import sqlite3


load_dotenv(override=True)

serper = GoogleSerperAPIWrapper()

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Search the Internet using Serper"
)


pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

def push_tool(text: str):
  
    requests.post(
        pushover_url,
        data={"token": pushover_token, "user": pushover_user, "message": text}
    )
    return "Push notification sent."

tool_push = Tool(
    name="send_push_notification",
    func=push_tool,
    description="Send a push notification to the user"
)

tools = [tool_search, tool_push]


class State(TypedDict):
    messages: Annotated[list, add_messages]      # conversation history
    plan: str | None                             # planner output
    worker_output: str | None                    # worker output
    evaluation: str | None                       # evaluator decision
    redo: bool | None                            # True = retry worker



llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)



def planner(state: State):
    """
    Extracts user message and creates a short execution plan.
    FIX: state["messages"][-1] is HumanMessage, so use .content (not ["content"])
    """
    user_message = state["messages"][-1].content

    prompt = f"""
You are a task planner.

User request: {user_message}

Break the task into a SHORT actionable plan.
No bullet points. Just a short text.
"""
    result = llm.invoke(prompt)
    return {"plan": result.content}



def worker(state: State):
    """
    Executes the plan.
    Uses llm_with_tools, so it can call search/push.
    FIX: Always return worker_output + append result to messages.
    """
    plan = state["plan"]

    prompt = [
        {"role": "system", "content": "You are a worker that executes tasks."},
        {"role": "user", "content": f"Follow this plan:\n{plan}"},
    ]

    result = llm_with_tools.invoke(prompt)

    return {
        "worker_output": result.content,
        "messages": [result]    # This keeps message history consistent
    }


def evaluator(state: State):
    """
    CRITICAL FIX:
    - Old version looked for "retry" in long LLM output → infinite loops.
    - Now LLM MUST output EXACTLY "retry" or "ok"
    """
    output = state["worker_output"]
    user_question = state["messages"][-1].content

    prompt = f"""
You are an evaluator.

User asked: {user_question}
Worker answered: {output}

Respond with EXACTLY one word:
"retry"  — if the worker answer is wrong/incomplete
"ok"     — if the worker answer is correct

NO explanations.
NO punctuation.
NO extra words.
"""

    result = llm.invoke(prompt)
    text = result.content.strip().lower()

    # FIX: ONLY accept exact match
    if text == "retry":
        return {"redo": True, "evaluation": text}

    # Default to OK even if LLM misbehaves
    return {"redo": False, "evaluation": "ok"}


graph_builder = StateGraph(State)

graph_builder.add_node("planner", planner)
graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "worker")

graph_builder.add_conditional_edges(
    "worker",
    tools_condition,
    {"tools": "tools", "__end__": "evaluator"}   # <-- FIX
)
# tools → back to worker
graph_builder.add_edge("tools", "worker")

# worker → evaluator
graph_builder.add_edge("worker", "evaluator")

# evaluator → retry or finish
graph_builder.add_conditional_edges(
    "evaluator",
    lambda state: "retry" if state["redo"] else "finish",
    {"retry": "worker", "finish": END}
)


conn = sqlite3.connect("day4_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)
config = {
    "configurable": {"thread_id": "day4-thread"},
    "recursion_limit": 100
}

def gradio_chat(message, history):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )

    return result["messages"][-1].content


ui = gr.ChatInterface(
    fn=gradio_chat,
    title="LangGraph Day-4 Agent (Fixed)"
)


if __name__ == "__main__":
    ui.launch()
