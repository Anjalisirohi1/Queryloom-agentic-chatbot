from langchain_core.messages import HumanMessage, AIMessage
from app.agent import nodes


class MockLLM:
    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return AIMessage(content="Mock response")


def test_trimming_removes_old_messages_and_keeps_latest(monkeypatch):
    mock_llm = MockLLM()

    monkeypatch.setattr(nodes, "llm", mock_llm)
    monkeypatch.setattr(nodes, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(
        nodes, "build_system_prompt", lambda state: "You are helpful."
    )

    old_text = "OLD HISTORY " * 200

    state = {
        "messages": [
            HumanMessage(content=old_text),
            AIMessage(content=old_text),
            HumanMessage(content="What is 2 + 2?"),
        ]
    }

    result = nodes.chat_node(state)

    sent_messages = mock_llm.received_messages

    assert sent_messages is not None
    assert any(
        m.type == "human" and m.content == "What is 2 + 2?"
        for m in sent_messages
    )
    assert not any(
        "OLD HISTORY" in str(m.content) for m in sent_messages
    )
    assert result["messages"][-1].content == "Mock response"

    print("PASS: Old messages trimmed; latest question preserved.")

def test_oversized_latest_message_returns_warning(monkeypatch):
    mock_llm = MockLLM()

    monkeypatch.setattr(nodes, "llm", mock_llm)
    monkeypatch.setattr(nodes, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(
        nodes, "build_system_prompt", lambda state: "You are helpful."
    )

    state = {
        "messages": [
            HumanMessage(content="VERY LONG QUESTION " * 500)
        ]
    }

    result = nodes.chat_node(state)

    assert "too large" in result["messages"][-1].content.lower()
    assert mock_llm.received_messages is None

    print("PASS: Oversized message rejected without calling LLM.")