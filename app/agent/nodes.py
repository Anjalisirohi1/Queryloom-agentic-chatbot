from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ChatState

load_dotenv()

MAX_INPUT_TOKENS = 6000

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    streaming=True,
)


def build_system_prompt(state: ChatState) -> str:
    """Build the prompt using both memory summaries."""

    parts = [SYSTEM_PROMPT]

    user_summary = state.get("summary", "")
    conversation_summary = state.get("conversation_summary", "")

    if user_summary and user_summary.strip() != "No durable memory recorded yet.":
        parts.append(
            "User facts and preferences:\n" + user_summary
        )

    if conversation_summary:
        parts.append(
            "Earlier conversation context:\n" + conversation_summary
        )

    return "\n\n".join(parts)


def _same_message(left, right) -> bool:
    """Compare messages using IDs when available."""

    if left.id is not None and right.id is not None:
        return left.id == right.id

    return left is right


def chat_node(state: ChatState):
    """Generate a response using the available context within budget."""

    messages = [
        SystemMessage(content=build_system_prompt(state)),
        *state["messages"],
    ]

    tokens_before = count_tokens_approximately(messages)

    trimmed_messages = trim_messages(
        messages,
        max_tokens=MAX_INPUT_TOKENS,
        token_counter="approximate",
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    tokens_after = count_tokens_approximately(trimmed_messages)

    print(
        f"\n[Token Trimming] Before: ~{tokens_before} tokens | "
        f"After: ~{tokens_after} tokens | "
        f"Budget: {MAX_INPUT_TOKENS}"
    )

    latest_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if message.type == "human"
        ),
        None,
    )

    if latest_user_message is not None:
        latest_was_kept = any(
            _same_message(message, latest_user_message)
            for message in trimmed_messages
        )

        if not latest_was_kept:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Your latest message is too large for the "
                            "current context budget. Please shorten it "
                            "and try again."
                        )
                    )
                ]
            }

    response = llm.invoke(trimmed_messages)

    return {"messages": [response]}