from langgraph.graph import StateGraph,START,END
from app.agent.state import ChatState
from app.agent.nodes import chat_node
from app.storage.checkpointer import checkpointer

builder = StateGraph(ChatState)

builder.add_node("chat_node",chat_node)

builder.add_edge(START,"chat_node")

builder.add_edge("chat_node",END)

graph = builder.compile(checkpointer=checkpointer)