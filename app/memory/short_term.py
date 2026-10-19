from langchain_core.messages import AnyMessage

from app.agent.nodes import llm

NO_DURABLE_MEMORY = "No durable memory recorded yet."


def _content_to_text(content) -> str:
    """Convert message content into plain text."""

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


def _messages_to_text(messages: list[AnyMessage]) -> str:
    """Format user and assistant messages for summarization."""

    conversation = []

    for message in messages:
        if message.type == "human":
            role = "User"
        elif message.type == "ai":
            role = "Assistant"
        else:
            continue

        content = _content_to_text(message.content)

        if content.strip():
            conversation.append(f"{role}: {content}")

    return "\n".join(conversation)


def _get_text_response(response) -> str:
    """Extract and validate text from an LLM response."""

    content = _content_to_text(response.content)

    if not content.strip():
        raise ValueError("Summarizer returned an empty response.")

    return content.strip()


def summarize_messages(
    existing_summary: str,
    messages: list[AnyMessage],
) -> str:
    """Summarize durable user facts, preferences, goals and corrections."""

    if existing_summary.strip() == NO_DURABLE_MEMORY:
        existing_summary = ""

    conversation_text = _messages_to_text(messages)

    if not conversation_text:
        return existing_summary or NO_DURABLE_MEMORY

    prompt = f"""
You maintain Queryloom's durable user memory.

Existing summary:
{existing_summary or "No durable user memory yet."}

Messages to consider:
{conversation_text}

Rules:
- Preserve facts and preferences explicitly stated or confirmed by the user.
- Preserve ongoing goals, projects, decisions and relevant next steps.
- Apply explicit corrections by replacing incorrect older values.
- Do not infer personal interests, skills, intentions or plans from questions.
- Do not treat assistant suggestions or claims as user facts.
- Do not turn aspirations into plans or suggestions into commitments.
- The messages may be only part of a longer conversation.
- Preserve useful existing facts even when absent from this chunk.
- Omit assistant questions and offers the user has not adopted.
- Remove repetition, small talk and irrelevant details.
- Never include credentials, API keys, passwords or access tokens.
- Keep the summary concise, ideally under 200 words.
- Use headings only when useful:
  User Facts and Preferences
  Projects and Goals
  Current Tasks and Decisions
  Open Questions and Next Steps
- Do not claim that information is absent merely because it is missing
  from this chunk.
- If no durable user information exists and no existing summary exists,
  return exactly: {NO_DURABLE_MEMORY}
- Return only the updated summary.
"""

    return _get_text_response(llm.invoke(prompt))


def summarize_conversation_context(
    existing_summary: str,
    messages: list[AnyMessage],
) -> str:
    """Summarize earlier conversation content for future follow-ups."""

    conversation_text = _messages_to_text(messages)

    if not conversation_text:
        if existing_summary:
            return existing_summary

        raise ValueError("No conversation content available to summarize.")

    prompt = f"""
You maintain Queryloom's running conversation-context summary.

Existing conversation summary:
{existing_summary or "No previous conversation context."}

New conversation messages:
{conversation_text}

Rules:
- Merge with the existing summary and avoid repetition.
- Preserve the user's questions, main topics and useful answers.
- For tutorials, retain the subject, major concepts, conclusions and
  enough detail to answer likely follow-up questions.
- For troubleshooting, preserve the problem, attempted fixes, errors,
  current implementation and unresolved issues.
- Preserve important technical details, decisions and unfinished
  user-requested tasks.
- Do not invent details or change the meaning of earlier messages.
- Do not turn assistant suggestions or questions into user commitments.
- This is a conversation summary, not just a user profile.
- Keep it concise but informative, ideally under 500 words.
- Return only the updated conversation-context summary.
"""

    return _get_text_response(llm.invoke(prompt))