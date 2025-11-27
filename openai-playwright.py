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

from playwright.sync_api import sync_playwright

# --------------------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------------------
load_dotenv(override=True)


# --------------------------------------------------------------
# 2. Tools (Search + Push Notification)
# --------------------------------------------------------------

# Serper Search Tool
serper = GoogleSerperAPIWrapper()

tool_search = Tool(
    name="search",
    func=serper.run,
    description="Search the Internet using Serper"
)

# Pushover Tool-
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"


def push(text: str):
    """Send Pushover notification."""
    requests.post(
        pushover_url,
        data={
            "token": pushover_token,
            "user": pushover_user,
            "message": text
        }
    )
    return "success"


tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Send a notification using Pushover"
)
NEWS_SITES = {
    "cnn": "https://edition.cnn.com/world",
    "cnn latest": "https://edition.cnn.com/world",
    "bbc": "https://www.bbc.com/news",
    "bbc latest": "https://www.bbc.com/news",
    "bbc news": "https://www.bbc.com/news",
}

def resolve_news_url(query: str):
    """Return the correct news URL for CNN/BBC."""
    q = query.lower()
    for key, url in NEWS_SITES.items():
        if key in q:
            return url
    return "UNKNOWN"

tool_news_resolver = Tool(
    name="resolve_news_url",
    func=resolve_news_url,
    description="Resolve BBC or CNN news queries into the correct URL. Input: plain text. Output: URL."
)

def web_browse(url: str):
    """Open a webpage and return HTML content."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000)
            page.wait_for_load_state("networkidle")
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        return f"Playwright error: {str(e)}"




tool_browse = Tool(
    name="web_browse",
    func=web_browse,
    description="Visit a webpage and return the full HTML content. Input must be a URL."
)



tools = [tool_search, tool_push, tool_browse, tool_news_resolver]
# --------------------------------------------------------------
# 3. State Definition
# --------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------------------
# 4. Build LangGraph
# --------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)


def chatbot(state: State):
    """Main LLM Node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# If model calls a tool → go to ToolNode
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    }
)
# After tool execution → go back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Base edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# --------------------------------------------------------------
# 5. Add SQLite Memory Checkpointing
# --------------------------------------------------------------
conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "day3-thread"}}




# --------------------------------------------------------------
# 7. Gradio Chat UI
# --------------------------------------------------------------

def gradio_chat(message, history):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )

    return result["messages"][-1].content


ui = gr.ChatInterface(
    fn=gradio_chat,
    title="LangGraph Day 3 Playright and Agent"
)



if __name__ == "__main__":
    ui.launch()
