from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_stream_writer
from uuid import uuid4
import re
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ChatState
from app.agent.tools import tools, search_knowledge_base

load_dotenv()

MAX_INPUT_TOKENS = 6000

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    streaming=True,
)

llm_with_tools = llm.bind_tools(tools)


def _content_to_text(content) -> str:
    """Convert a message content payload into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else item.get("text", "")
            for item in content
            if isinstance(item, str)
            or (isinstance(item, dict) and item.get("type") == "text")
        )
    return ""


def _display_document_name(source: str) -> str:
    """Make an indexed source name readable for user-facing summaries."""
    name = source.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"-[0-9a-fA-F]{8}(?=\.[^.]+$)", "", name)
    name = re.sub(r"\.[^.]+$", "", name)
    return name.replace("_", " ").strip() or "Uploaded document"


def _is_multi_document_summary_request(
    question: str,
    active_documents: list[str],
) -> bool:
    """Recognize explicit requests to summarize multiple active documents."""
    if len(active_documents) < 2:
        return False
    q = " ".join(question.casefold().split())
    asks_for_summary = re.search(
        r"\b(?:summar(?:y|ies|ize|ise|izing|ising)|overview|brief)\b", q
    )
    # Plural nouns mean the user asked for multiple files. Singular "the document"
    # is ambiguous when multiple files are active and must not summarize all of them.
    refers_to_multiple_documents = re.search(r"\b(?:pdfs|documents|files)\b", q)
    return bool(asks_for_summary and refers_to_multiple_documents)


def _is_ambiguous_single_document_summary_request(
    question: str,
    active_documents: list[str],
) -> bool:
    """Detect a generic singular summary request when multiple files are active."""
    if len(active_documents) < 2:
        return False

    q = " ".join(question.casefold().split())
    if not re.search(
        r"\b(?:summar(?:y|ies|ize|ise|izing|ising)|overview|brief)\b", q
    ):
        return False
    if re.search(r"\b(?:pdfs|documents|files|both|all|each|every)\b", q):
        return False

    question_tokens = set(re.findall(r"[a-z0-9]+", q))
    generic_tokens = {"the", "this", "that", "uploaded", "attached", "current", "pdf", "document", "file"}
    matching_sources = []
    for source in active_documents:
        source_tokens = set(re.findall(r"[a-z0-9]+", _display_document_name(source).casefold()))
        source_tokens -= generic_tokens
        if source_tokens & question_tokens:
            matching_sources.append(source)

    if len(matching_sources) == 1:
        return False

    return bool(re.search(
        r"\b(?:(?:the|this|that|uploaded|attached|current)\s+)?(?:pdf|document|file)\b", q
    ))


def _user_explicitly_asks_about_capabilities(question: str) -> bool:
    """Only preserve a capability list when the user actually asks for one."""
    q = question.lower()
    patterns = (
        r"\bwhat can (?:you|queryloom) do\b",
        r"\bwhat tasks can (?:you|queryloom) perform\b",
        r"\blist (?:your|queryloom's) capabilities\b",
        r"\bwhat are (?:your|queryloom's) capabilities\b",
        r"\bwhat can i ask (?:you|queryloom)\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def _remove_unrequested_sections(text: str, question: str) -> str:
    """Remove common off-topic capability/news sections from a document answer.

    This is a presentation guard, not a replacement answer: the LLM still writes
    the answer and follow-up suggestions. It prevents a known, repeated failure
    mode where the model appends its generic capabilities or unrelated topics.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if _user_explicitly_asks_about_capabilities(question):
        return text.strip()


    suggestion_match = re.search(
        r"(?im)^\s*(?:\*{1,2}|_{1,2})?\s*you could also ask\s*:?(?:\*{1,2}|_{1,2})?\s*$",
        text,
    )
    if suggestion_match:
        main_text = text[:suggestion_match.start()]
        suggestion_text = text[suggestion_match.start():]
    else:
        main_text = text
        suggestion_text = ""

    unwanted_heading = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*{1,2}|_{1,2})?\s*(?:"
        r"concise list of tasks i can perform|"
        r"list of tasks i can perform|"
        r"what i can help you with(?:\s*\([^\n)]*\))?|"
        r"what i can help with|"
        r"available capabilities|"
        r"things i can help with|"
        r"what(?:'|’)s going on in india recently\??|"
        r"what is going on in india recently\??|"
        r"current events(?: and news)?"
        r")\s*(?:\*{1,2}|_{1,2})?\s*:?[ \t]*$"
    )
    match = unwanted_heading.search(main_text)
    if match:
        main_text = main_text[:match.start()].rstrip()

    main_text = re.sub(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*{1,2})?note(?:\*{1,2})?\s*:\s*"
        r"(?:i(?:'|’)m unable to|i am unable to|i cannot)\s+"
        r"(?:clear|forget)[^\n]*\n?",
        "",
        main_text,
    ).rstrip()

    result = main_text
    if suggestion_text:
        result = f"{main_text}\n\n{suggestion_text.strip()}" if main_text else suggestion_text.strip()
    return result.strip()


def _build_document_answer_prompt(
    active_documents: list[str],
    *,
    no_evidence_found: bool,
    document_needs_clarification: bool,
) -> str:
    """Use a focused prompt for the final document answer, without shared memories."""
    documents = "\n".join(f"- {_display_document_name(name)}" for name in active_documents) or "- (not specified)"
    instructions = [
        """You are Queryloom, a reliable assistant. Answer only the user's latest question,
        using the retrieved document evidence provided below. Do not call or request tools in
        this final-answer step.

        ## Current document-answer turn: strict scope
        Answer only the user's latest question, using the retrieved document evidence in this
        conversation. Do not answer earlier questions from other conversations. The user's
        cross-thread memory and earlier conversation summaries are intentionally not included
        in this final document-answer prompt; do not invent or mention them.

        For a substantive answer, write a concise direct answer, then add exactly one section
        headed `You could also ask:` with 2 or 3 short Markdown bullet questions. Suggestions
        must be supported by the document evidence. Do not add a list of Queryloom's capabilities,
        unrelated news or topics, generic notes about memory/history, or an offer of unrelated
        services. Never expose generated source IDs or raw indexed filenames or extensions.
        For a request to summarize multiple PDFs, use only the clean, human-readable headings
        created for the document result blocks. Summarize each PDF separately
        under a short, human-readable heading. Use only evidence from that PDF's labeled result.
        If a PDF has no relevant retrieved evidence, state that this PDF could not be summarized
        from the retrieved content; still summarize other PDFs when evidence is available. Do not
        mention internal implementation details such as tools, CRAG, retrieval, chunks, or the
        knowledge base. The user-facing answer and suggestions must be in natural wording.""",
        "Uploaded document(s) for this turn:\n" + documents,
    ]
    if no_evidence_found:
        instructions.append(
            "The search did not find evidence for the exact fact requested. Say naturally "
            "that the requested detail is not stated in the uploaded document. Do not "
            "guess, infer a number, or suggest that the missing fact is available. "
            "Still provide 2 or 3 relevant follow-up questions grounded in other content "
            "actually supported by this document."
        )
    if document_needs_clarification:
        instructions.append(
            "The intended uploaded document could not be identified uniquely. Ask one "
            "concise clarifying question only. Do not add suggestions or other sections."
        )
    return "\n\n".join(instructions)

def build_system_prompt(state: ChatState) -> str:
    """Build the prompt using both memory summaries."""

    parts = [
        SYSTEM_PROMPT,
        """## Current-turn relevance rules
        Answer the user's latest message, not an earlier question from conversation history.
        Treat long-term memory and earlier conversation summaries only as background context;
        they are not requests to answer again. Mention information from them only when it is
        relevant to the latest request. Do not introduce unrelated topics, unsolicited news,
        or a generic list of capabilities. For substantive requests, any follow-up suggestions
        must stay on the current topic and be supported by the available context. Do not expose
        internal memory, routing, retrieval, CRAG, or tool implementation details.""",
    ]

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
    """Compare messages by ID, falling back to role and content if IDs are absent."""

    if left.id is not None and right.id is not None:
        return left.id == right.id

    return left.type == right.type and left.content == right.content


def should_retrieve_documents(state: ChatState, active_documents: list[str]) -> bool:
    """Determine whether the latest user request needs uploaded-document retrieval."""

    if not active_documents:
        return False

    latest_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if message.type == "human"
        ),
        None,
    )

    if latest_user_message is None:
        return False

    def content_to_text(content):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(
                item if isinstance(item, str)
                else item.get("text", "")
                for item in content
                if isinstance(item, str)
                or (isinstance(item, dict) and item.get("type") == "text")
            )

        return ""

    latest_question = content_to_text(
        latest_user_message.content
    ).strip()

    if not latest_question:
        return False


    if _is_multi_document_summary_request(latest_question, active_documents):
        return True

    recent_context = []

    for message in state["messages"][-8:]:
        if message.type not in {"human", "ai"}:
            continue

        content = content_to_text(message.content).strip()

        if content:
            role = "User" if message.type == "human" else "Assistant"
            recent_context.append(f"{role}: {content[:500]}")

    prompt = f"""
    You are a routing classifier for an AI assistant.

    Available uploaded documents:
    {chr(10).join(f"- {name}" for name in active_documents)}

    Recent conversation:
    {chr(10).join(recent_context)[-3000:]}

    Latest user request:
    {latest_question}

    Classify the latest request:

    DOCUMENT: Answering requires consulting an uploaded document.
    This includes summaries, questions about document facts, and
    follow-up references such as "this PDF" when conversation context
    connects them to an uploaded document.

    GENERAL: The request is unrelated to uploaded documents and can
    be handled without retrieving their contents.

    If multiple documents are available and the intended document
    is unclear, classify as GENERAL so the assistant can clarify first.

    Return exactly DOCUMENT or GENERAL.
    """

    response = llm.invoke(prompt)
    decision = content_to_text(response.content).strip().upper()
    decision = decision.splitlines()[0].strip("` .:")

    return decision == "DOCUMENT"

def chat_node(state: ChatState, config: RunnableConfig):
    """Generate a response using the available context within budget."""

    system_prompt = build_system_prompt(state)
    system_prompt += (
        "\n\n## Follow-up Suggestions\n"
        "After answering a substantive user request, add a short section "
        "headed `You could also ask:` with 2 or 3 concise follow-up questions. "
        "Write them as ordinary Markdown bullets, not clickable controls. "
        "Make them relevant to the user's question and the conversation. "
        "For uploaded documents, suggest questions grounded in information "
        "actually present in, or supported by, the available document context; "
        "do not imply that an unmentioned fact is available. Match the user's "
        "language. Do not add a generic capabilities list or a vague offer "
        "of help. Skip suggestions after greetings, acknowledgements, refusals, "
        "or clarifying questions that require the user to answer first. "
        "Generate the answer and suggestions in this same response; do not "
        "call another tool or make a separate generation for suggestions."
    )
    active_documents = config.get("configurable", {}).get(
        "active_documents", []
    )


    latest_request = next(
        (message for message in reversed(state["messages"]) if message.type == "human"),
        None,
    )
    latest_request_text = (
        _content_to_text(latest_request.content).strip()
        if latest_request is not None
        else ""
    )
    if _is_ambiguous_single_document_summary_request(
        latest_request_text, active_documents
    ):
        choices = "\n".join(
            f"- {_display_document_name(name)}" for name in active_documents
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Which document would you like me to summarize?\n\n"
                        + choices
                    )
                )
            ]
        }


    latest_human_index = next(
        (i for i in range(len(state["messages"]) - 1, -1, -1)
         if state["messages"][i].type == "human"),
        -1,
    )
    previous_message = (
        state["messages"][latest_human_index - 1]
        if latest_human_index > 0 else None
    )
    has_tool_result_for_selection = (
        latest_human_index >= 0
        and any(
            message.type == "tool"
            and getattr(message, "name", None) == "search_knowledge_base"
            for message in state["messages"][latest_human_index + 1:]
        )
    )
    selected_document_source = None
    is_answer_to_document_choice = (
        len(active_documents) > 1
        and previous_message is not None
        and previous_message.type == "ai"
        and "which document would you like me to summarize?"
        in _content_to_text(previous_message.content).casefold()
    )
    if is_answer_to_document_choice:
        selected_text = " ".join(
            re.findall(r"[a-z0-9]+", latest_request_text.casefold())
        )
        selected_sources = [
            source for source in active_documents
            if selected_text == " ".join(
                re.findall(r"[a-z0-9]+", _display_document_name(source).casefold())
            )
        ]
        if len(selected_sources) == 1:
            selected_document_source = selected_sources[0]
            if not has_tool_result_for_selection:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "search_knowledge_base",
                                    "args": {
                                        "query": (
                                            "Summarize this document, covering its purpose, "
                                            "key topics, main sections, and important conclusions."
                                        ),
                                        "source": selected_document_source,
                                    },
                                    "id": str(uuid4()),
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }

    if active_documents:
        document_list = "\n".join(
            f"- {name}" for name in active_documents
        )

        system_prompt += (
            "\n\nSuccessfully indexed documents available in this conversation:\n"
            f"{document_list}\n\n"
            "When the user asks about an uploaded document or refers to it "
            "indirectly, such as 'this PDF' or 'summarize this', use the "
            "search_knowledge_base tool before answering. "
            "When calling search_knowledge_base, pass the exact indexed "
            "source filename from the available documents list as the "
            "'source' argument. Use the user's request to construct the "
            "'query' argument. If multiple documents are listed and the "
            "intended document is unclear, ask which one the user means."
        )

    messages = [
        SystemMessage(content=system_prompt),
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

    latest_user_index = next(
        (
            i
            for i in range(len(state["messages"]) - 1, -1, -1)
            if state["messages"][i].type == "human"
        ),
        -1,
    )

    latest_user_message = (
        state["messages"][latest_user_index]
        if latest_user_index >= 0
        else None
    )
    if latest_user_message is not None:
        trimmed_latest_user = next(
            (message for message in reversed(trimmed_messages) if message.type == "human"),
            None,
        )
        latest_was_kept = (
            trimmed_latest_user is not None
            and _same_message(trimmed_latest_user, latest_user_message)
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


    messages_after_latest_user = (
        state["messages"][latest_user_index + 1:]
        if latest_user_index >= 0
        else []
    )
    kb_tool_results = [
        message
        for message in messages_after_latest_user
        if message.type == "tool"
        and getattr(message, "name", None) == "search_knowledge_base"
    ]
    kb_tool_result = kb_tool_results[-1] if kb_tool_results else None

    result_text = ""
    no_evidence_found = False
    document_needs_clarification = False

    if kb_tool_results:

        source_by_tool_call_id = {}
        for message in messages_after_latest_user:
            if message.type != "ai":
                continue
            for call in getattr(message, "tool_calls", []) or []:
                if call.get("name") == "search_knowledge_base":
                    source_by_tool_call_id[call.get("id")] = (
                        call.get("args", {}).get("source", "")
                    )

        result_blocks = []
        raw_result_texts = []
        for tool_result in kb_tool_results:
            content = tool_result.content
            if isinstance(content, str):
                tool_result_text = content
            elif isinstance(content, list):
                tool_result_text = " ".join(
                    item if isinstance(item, str) else item.get("text", "")
                    for item in content
                    if isinstance(item, str)
                    or (isinstance(item, dict) and item.get("type") == "text")
                )
            else:
                tool_result_text = str(content)


            tool_result_text = re.sub(
                r"(?m)^Source: [^\n]*\n?", "", tool_result_text
            ).strip()
            raw_result_texts.append(tool_result_text)
            source_name = source_by_tool_call_id.get(
                getattr(tool_result, "tool_call_id", None), ""
            )
            label = _display_document_name(source_name) if source_name else "Uploaded document"
            result_blocks.append(f"Document: {label}\n{tool_result_text}")

        result_text = "\n\n---\n\n".join(result_blocks)
        latest_question_for_scope = (
            _content_to_text(latest_user_message.content)
            if latest_user_message is not None
            else ""
        )
        is_multi_document_summary = _is_multi_document_summary_request(
            latest_question_for_scope, active_documents
        )
        no_evidence_found = (
            not is_multi_document_summary
            and bool(raw_result_texts)
            and all(
                "no relevant information found in the knowledge base" in text.lower()
                for text in raw_result_texts
            )
        )
        document_needs_clarification = any(
            "could not uniquely identify the requested document" in text.lower()
            for text in raw_result_texts
        )

    model_messages = trimmed_messages
    if kb_tool_result is not None:
        focused_prompt = _build_document_answer_prompt(
            active_documents,
            no_evidence_found=no_evidence_found,
            document_needs_clarification=document_needs_clarification,
        )
        if selected_document_source:
            focused_prompt += (
                "\n\nThe user selected this document after being asked which one to summarize: "
                f"{_display_document_name(selected_document_source)}. "
                "Summarize only this selected document using its retrieved evidence."
            )
        if is_multi_document_summary:
            focused_prompt += (
                "\n\nThis is a summary of multiple uploaded documents. Write a separate, "
                "clearly titled summary for every requested document, using only the evidence "
                "inside that document's labeled block. Do not omit a document. If a document's "
                "block says no relevant information was found, state that limitation for that "
                "document rather than inventing a summary."
            )
        current_question_for_summary = (
            _content_to_text(latest_user_message.content)
            if latest_user_message is not None
            else ""
        )
        is_document_summary = (
            selected_document_source is not None
            or is_multi_document_summary
            or bool(re.search(r"\b(?:summar(?:y|ies|ize|ise|izing|ising)|overview)\b", current_question_for_summary, re.I))
        )
        if is_document_summary:
            focused_prompt += (
                "\n\nSUMMARY COVERAGE REQUIREMENTS: Give a document-level summary, not a summary "
                "of just one retrieved excerpt. Cover every distinct section/topic supported by "
                "the supplied evidence. When the evidence includes numbered programs, experiments, "
                "or algorithms, enumerate each distinct one and briefly state what it covers. "
                "Include purpose and key takeaways where the evidence supports them. Do not let "
                "one code snippet or sample output dominate the answer when other sections are "
                "present. Do not invent facts or sections. If the supplied evidence contains only "
                "one section and cannot support a whole-document summary, explicitly say the "
                "retrieved evidence is partial rather than presenting it as complete."
            )
        focused_prompt += (
            "\n\nRetrieved document evidence from the current search:\n"
            + result_text
        )
        clean_history = []
        for message in trimmed_messages:
            if isinstance(message, SystemMessage):
                continue
            if message.type == "tool":
                continue
            if message.type == "ai" and getattr(message, "tool_calls", None):
                continue
            clean_history.append(message)
        model_messages = [SystemMessage(content=focused_prompt), *clean_history]


    # Check whether a tool has already returned a result this turn.
    tool_used_this_turn = (
        latest_user_index >= 0
        and any(
            message.type == "tool"
            for message in messages_after_latest_user
        )
    )

    should_force_retrieval = (
        bool(active_documents)
        and not tool_used_this_turn
        and should_retrieve_documents(state, active_documents)
    )

    latest_question_text = (
        _content_to_text(latest_user_message.content)
        if latest_user_message is not None
        else ""
    )
    if should_force_retrieval and _is_multi_document_summary_request(
        latest_question_text, active_documents
    ):
        retrieval_query = (
            "Summarize this document, covering its purpose, key topics, "
            "main sections, and important conclusions."
        )
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge_base",
                            "args": {"query": retrieval_query, "source": document_name},
                            "id": str(uuid4()),
                            "type": "tool_call",
                        }
                        for document_name in active_documents
                    ],
                )
            ]
        }


    if should_force_retrieval and len(active_documents) == 1:
        if isinstance(latest_user_message.content, str):
            retrieval_query = latest_user_message.content
        else:
            retrieval_query = str(latest_user_message.content)

        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge_base",
                            "args": {
                                "query": retrieval_query,
                                "source": active_documents[0],
                            },
                            "id": str(uuid4()),
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    if kb_tool_result is not None:
        model = llm
    elif should_force_retrieval:
        model = llm.bind_tools(
            [search_knowledge_base],
            tool_choice="search_knowledge_base",
        )
    else:
        model = llm_with_tools

    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None

    full_chunk = None

    for chunk in model.stream(model_messages):
        content = chunk.content

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
            )
        else:
            text = ""

        if writer is not None and text:
            writer({"token": text})

        full_chunk = (
            chunk if full_chunk is None else full_chunk + chunk
        )

    if full_chunk is not None:
        response_content = full_chunk.content

        if not full_chunk.tool_calls:
            response_text = _content_to_text(response_content)
            latest_question = (
                _content_to_text(latest_user_message.content)
                if latest_user_message is not None
                else ""
            )
            response_content = _remove_unrequested_sections(
                response_text,
                latest_question,
            )
        response = AIMessage(
            content=response_content,
            tool_calls=full_chunk.tool_calls,
        )
    else:
        response = AIMessage(content="No response generated.")

    return {"messages": [response]}
