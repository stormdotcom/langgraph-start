import os
import sqlite3
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

load_dotenv(override=True)

system_prompt = """
You are an agent with access to tools.
You must call exactly one tool at a time.
Never call multiple tools in one assistant message.
If a task requires multiple steps, call one tool, wait for the tool result, then continue.
Respond normally only when no tool is needed.
"""

serper = GoogleSerperAPIWrapper()
tool_search = Tool(
    name="search",
    func=serper.run,
    description="Search the Internet"
)

pushover_url = "https://api.pushover.net/1/messages.json"
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")

def push(text: str):
    requests.post(pushover_url, data={"token": pushover_token, "user": pushover_user, "message": text})
    return "success"

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Send a push notification"
)

def write_file(data: dict):
    with open(data["path"], "w") as f:
        f.write(data["content"])
    return "file saved"

tool_write_file = Tool(
    name="write_file",
    func=write_file,
    description="Write a file to disk"
)

tools = [tool_search, tool_push, tool_write_file]

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)

def chatbot(state: State):
    msgs = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": END}
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "day5-thread"}}

def gradio_chat(message, history):
    result = graph.invoke(
    {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    },
    config=config
)
    return result["messages"][-1].content

ui = gr.ChatInterface(fn=gradio_chat, title="LangGraph Level 4 Agent")

if __name__ == "__main__":
    ui.launch()
