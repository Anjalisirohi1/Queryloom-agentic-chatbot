from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.agent.tools import tools

from app.agent.state import ChatState
from app.agent.nodes import chat_node
from app.memory.memory_manager import memory_manager_node
from app.memory.long_term import (
    load_long_term_memory_node,
    save_long_term_memory_node,
)
from app.storage.checkpointer import checkpointer
from app.storage.postgres_store import store

builder = StateGraph(ChatState)

builder.add_node("load_long_term_memory", load_long_term_memory_node)

builder.add_node("memory_manager", memory_manager_node)

builder.add_node("save_long_term_memory", save_long_term_memory_node)

builder.add_node("tools",ToolNode(tools))

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "load_long_term_memory")
builder.add_edge("load_long_term_memory", "memory_manager")
builder.add_edge("memory_manager", "save_long_term_memory")
builder.add_edge("save_long_term_memory", "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)