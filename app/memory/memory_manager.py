import logging

from langchain_core.messages import RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.agent.nodes import MAX_INPUT_TOKENS, build_system_prompt, llm
from app.agent.state import ChatState
from app.memory.short_term import (
    NO_DURABLE_MEMORY,
    summarize_conversation_context,
    summarize_messages,
)

logger = logging.getLogger(__name__)

SUMMARY_TRIGGER = 10
KEEP_RECENT_MESSAGES = 5


def content_to_text(content) -> str:
    """Convert message content into readable text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return ""


def should_update_memory(state: ChatState) -> bool:
    """Detect meaningful user facts and memory corrections."""

    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")

    latest_user_message = next(
        (
            message
            for message in reversed(messages)
            if message.type == "human"
        ),
        None,
    )

    if latest_user_message is None:
        return False

    latest_text = content_to_text(
        latest_user_message.content
    ).strip()

    if not latest_text:
        return False

    recent_messages = []

    for message in messages[-6:]:
        if message.type == "human":
            role = "User"
        elif message.type == "ai":
            role = "Assistant"
        else:
            continue

        text = content_to_text(message.content)

        if text.strip():
            recent_messages.append(f"{role}: {text[:700]}")

    recent_context = "\n".join(recent_messages)

    classification_prompt = f"""
    You are Queryloom's memory-update classifier.

    Existing user memory:
    {existing_summary or NO_DURABLE_MEMORY}

    Recent conversation:
    {recent_context}

    Latest user message:
    {latest_text[:2000]}

    Return exactly UPDATE or NO_UPDATE.

    Choose UPDATE when the latest user message states or corrects a durable
    user fact, preference, goal, plan, decision or ongoing project.
    Understand meaning and context; do not rely on fixed keywords.

    Choose NO_UPDATE for greetings, small talk and general questions that
    introduce no durable user information.

    A question about a topic does not itself establish personal interest
    or expertise. Assistant suggestions are not user decisions unless
    the user explicitly adopts them.

    Return only the label.
    """

    response = llm.invoke(classification_prompt)
    decision_text = content_to_text(response.content).strip()

    if not decision_text:
        raise ValueError("Memory classifier returned an empty response.")

    decision = (
        decision_text.splitlines()[0]
        .strip()
        .strip("`")
        .upper()
        .rstrip(" .")
    )

    if decision not in {"UPDATE", "NO_UPDATE"}:
        raise ValueError(
            f"Unexpected memory classification: {decision_text!r}"
        )

    return decision == "UPDATE"


def validate_summary(summary: str) -> str:
    """Reject empty summaries before changing graph state."""

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(
            "Empty summary generated; messages must not be deleted."
        )

    return summary.strip()


def durable_user_messages(messages):
    """Only pass user-authored messages to durable-memory summarization."""

    return [
        message
        for message in messages
        if message.type == "human"
    ]


def complete_turn_chunk(messages):
    """Select older complete turns, ending at an assistant response."""

    chunk = list(messages[:-KEEP_RECENT_MESSAGES])

    while chunk and chunk[-1].type != "ai":
        chunk.pop()

    return chunk


def remove_messages(messages):
    """Build deletion operations only for messages with valid IDs."""

    if any(not message.id for message in messages):
        raise ValueError("Cannot safely delete messages without IDs.")

    return [
        RemoveMessage(id=message.id)
        for message in messages
    ]


def memory_manager_node(state: ChatState):
    """Maintain user memory and compact conversation safely."""

    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")
    existing_context = state.get("conversation_summary", "")

    if not messages:
        return {}


    estimated_tokens = count_tokens_approximately(
        [
            SystemMessage(content=build_system_prompt(state)),
            *messages,
        ]
    )

    if estimated_tokens > MAX_INPUT_TOKENS:
        updated_user_summary = None


        try:
            if should_update_memory(state):
                updated_user_summary = validate_summary(
                    summarize_messages(
                        existing_summary,
                        durable_user_messages(messages),
                    )
                )
        except Exception:
            logger.exception(
                "Memory update failed during token-pressure handling. "
                "Preserving messages to avoid losing context."
            )
            return {}

        messages_to_compact = complete_turn_chunk(messages)

        if not messages_to_compact:
            if updated_user_summary is not None:
                return {"summary": updated_user_summary}
            return {}

        new_context = validate_summary(
            summarize_conversation_context(
                existing_context,
                messages_to_compact,
            )
        )

        result = {
            "conversation_summary": new_context,
            "messages": remove_messages(messages_to_compact),
        }

        if updated_user_summary is not None:
            result["summary"] = updated_user_summary

        return result

    try:
        update_requested = should_update_memory(state)
    except Exception:
        logger.exception(
            "Memory classification failed; skipping this memory update."
        )
        update_requested = False

    updated_user_summary = None

    if update_requested:
        updated_user_summary = validate_summary(
            summarize_messages(
                existing_summary,
                durable_user_messages(messages),
            )
        )

    if len(messages) > SUMMARY_TRIGGER:
        messages_to_compact = complete_turn_chunk(messages)

        if not messages_to_compact:
            if updated_user_summary is not None:
                return {"summary": updated_user_summary}
            return {}

        new_context = validate_summary(
            summarize_conversation_context(
                existing_context,
                messages_to_compact,
            )
        )


        result = {
            "conversation_summary": new_context,
            "messages": remove_messages(messages_to_compact),
        }

        if updated_user_summary is not None:
            result["summary"] = updated_user_summary

        return result

    if updated_user_summary is not None:
        return {"summary": updated_user_summary}

    return {}