from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.agent.state import ChatState
from app.agent import nodes
from app.memory import memory_manager as mm


class MockLLM:
    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="Mock response")


def test_message_count_summarization_and_deletion(monkeypatch):
    monkeypatch.setattr(mm, "MAX_INPUT_TOKENS", 10000)
    monkeypatch.setattr(mm, "build_system_prompt", lambda state: "Helpful.")
    monkeypatch.setattr(mm, "should_update_memory", lambda state: False)
    monkeypatch.setattr(
        mm, "summarize_conversation_context",
        lambda old, messages: "Earlier conversation summary."
    )
    monkeypatch.setattr(
        mm, "summarize_messages",
        lambda old, messages: "Saved user facts."
    )

    messages = []
    for i in range(6):
        messages.extend([
            HumanMessage(content=f"Question {i}", id=f"h{i}"),
            AIMessage(content=f"Answer {i}", id=f"a{i}"),
        ])

    updates = mm.memory_manager_node({"messages": messages})

    remaining = add_messages(messages, updates["messages"])

    assert updates["conversation_summary"] == "Earlier conversation summary."
    assert updates["summary"] == "Saved user facts."
    assert len(remaining) < len(messages)
    assert all(m.id not in {f"h{i}" for i in range(3)} for m in remaining)

    print("PASS: Message-count summarization and deletion.")


def test_token_pressure_compaction_and_chat(monkeypatch):
    mock_llm = MockLLM()

    monkeypatch.setattr(
    mm, "should_update_memory", lambda state: False
    )
    monkeypatch.setattr(mm, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(nodes, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(
        mm, "build_system_prompt", lambda state: "Helpful."
    )
    monkeypatch.setattr(
        mm, "summarize_conversation_context",
        lambda old, messages: "Short earlier-context summary."
    )
    monkeypatch.setattr(
        nodes, "build_system_prompt",
        lambda state: "Helpful. " + state.get("conversation_summary", "")
    )
    monkeypatch.setattr(nodes, "llm", mock_llm)

    builder = StateGraph(ChatState)
    builder.add_node("memory_manager", mm.memory_manager_node)
    builder.add_node("chat_node", nodes.chat_node)
    builder.add_edge(START, "memory_manager")
    builder.add_edge("memory_manager", "chat_node")
    builder.add_edge("chat_node", END)
    test_graph = builder.compile()

    messages = [
        HumanMessage(content="OLD_CONTEXT_MARKER " * 300, id="old-h1"),
        AIMessage(content="OLD_CONTEXT_MARKER " * 300, id="old-a1"),
        HumanMessage(content="OLD_CONTEXT_MARKER " * 300, id="old-h2"),
        AIMessage(content="OLD_CONTEXT_MARKER " * 300, id="old-a2"),
        HumanMessage(content="Recent question 1", id="recent-h1"),
        AIMessage(content="Recent answer 1", id="recent-a1"),
        HumanMessage(content="Recent question 2", id="recent-h2"),
        AIMessage(content="Recent answer 2", id="recent-a2"),
        HumanMessage(content="Explain photosynthesis.", id="latest-user"),
    ]

    result = test_graph.invoke({"messages": messages})
    active = result["messages"]

    assert result["conversation_summary"] == "Short earlier-context summary."
    assert len(active) == 6  # 5 retained messages + new AI response
    assert all("OLD_CONTEXT_MARKER" not in str(m.content) for m in active)
    assert any(m.id == "latest-user" for m in active)
    assert active[-1].content == "Mock response"

    print("PASS: Summarization, deletion, recent messages, and chat.")

def test_token_pressure_updates_both_summaries(monkeypatch):
    monkeypatch.setattr(mm, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(mm, "build_system_prompt", lambda state: "Helpful.")
    monkeypatch.setattr(mm, "should_update_memory", lambda state: True)

    monkeypatch.setattr(
        mm, "summarize_messages",
        lambda old, messages: "User name: Anjali"
    )
    monkeypatch.setattr(
        mm, "summarize_conversation_context",
        lambda old, messages: "Earlier conversation summary."
    )

    messages = []
    for i in range(6):
        messages.extend([
            HumanMessage(
                content=f"Conversation context {i} " * 100,
                id=f"h{i}",
            ),
            AIMessage(
                content=f"Assistant response {i} " * 100,
                id=f"a{i}",
            ),
        ])

    updates = mm.memory_manager_node({"messages": messages})

    assert updates["summary"] == "User name: Anjali"
    assert updates["conversation_summary"] == "Earlier conversation summary."
    assert updates["messages"], "Old messages should be removed."

    print("PASS: Both summaries updated during token pressure.")