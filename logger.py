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

tools = [tool_search, tool_push]




class State(TypedDict):
    messages: Annotated[list, add_messages]


def logger(state: State):
    print("\n [STATE UPDATE]")
    print(state)
    return state

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)


def chatbot(state: State):
    """Main LLM Node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("logger", logger)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))


graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    }
)

graph_builder.add_edge("tools", "chatbot")

# Base edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "logger")
graph_builder.add_edge("logger", END)



conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "day3-thread"}}




def cli_chat():
    print("LangGraph CLI. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )

        ai_reply = result["messages"][-1].content
        print("AI:", ai_reply)




def gradio_chat(message, history):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config
    )

    return result["messages"][-1].content


ui = gr.ChatInterface(
    fn=gradio_chat,
    title="LangGraph Day 3 Agent"
)



if __name__ == "__main__":
    ui.launch()
