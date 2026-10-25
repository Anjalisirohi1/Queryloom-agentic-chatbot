import hashlib
import html
import inspect
import logging
import re
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from streamlit_cookies_controller import CookieController
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Ensure project imports work when Streamlit is launched from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from app.agent.graph import graph
from app.agent.nodes import llm
from app.rag.ingestion import ingest_document
from app.storage.checkpointer import checkpointer
from app.storage.postgres_store import store


# ---------- Configuration ----------
st.set_page_config(
    page_title="Queryloom",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Full exception details go to a log file; the terminal gets concise messages.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_PATH = LOG_DIR / "queryloom_errors.log"
error_file_logger = logging.getLogger("queryloom.error_file")
error_file_logger.setLevel(logging.ERROR)
error_file_logger.propagate = False
if not any(
    isinstance(handler, logging.FileHandler)
    and Path(handler.baseFilename) == ERROR_LOG_PATH.resolve()
    for handler in error_file_logger.handlers
):
    error_file_handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    error_file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    error_file_logger.addHandler(error_file_handler)

# Persistent anonymous identity for each browser.
COOKIE_NAME = "queryloom_user_id"
cookie_controller = CookieController(key="queryloom_cookie_controller")

if "queryloom_user_id" in st.session_state:
    USER_ID = st.session_state["queryloom_user_id"]
else:
    USER_ID = st.context.cookies.get(COOKIE_NAME)

    if not isinstance(USER_ID, str) or not USER_ID.startswith("user-"):
        USER_ID = f"user-{uuid4().hex}"

        cookie_controller.set(
            COOKIE_NAME,
            USER_ID,
            max_age=60 * 60 * 24 * 365,
            secure=st.context.url.startswith("https://"),
            same_site="lax",
        )

    st.session_state["queryloom_user_id"] = USER_ID

THREAD_NAMESPACE = ("queryloom", "users", USER_ID, "threads")
DEFAULT_TITLE = "New chat"
UPLOAD_TITLE = "Uploaded documents"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ASSISTANT_AVATAR = "✳"

# Streamlit renamed `use_container_width` to `width="stretch"` in newer
# releases. Pick whichever the installed version supports.
if "width" in inspect.signature(st.button).parameters:
    FULL_WIDTH = {"width": "stretch"}
else:
    FULL_WIDTH = {"use_container_width": True}


# ---------- Styling ----------
# Layout rule: the sidebar keeps Streamlit's native width and collapse
# behaviour. The conversation and the composer share one centred, max-width
# column, so collapsing or reopening the sidebar reflows the page symmetrically
# and nothing is positioned with fixed offsets.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #0c1322;
        --raised: #111a2b;
        --hover: #151f33;
        --line: #26334a;
        --line-soft: #1b273d;
        --text: #edf2fa;
        --muted: #91a0b8;
        --lime: #b7f36b;
        --teal: #63dfc3;
    }

    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        background: radial-gradient(ellipse at 80% 0%, #19352f 0%, transparent 36%), var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    /* Translucent header: the native sidebar toggle stays visible and
       scrolling messages never collide with the toolbar text. */
    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.82);
        backdrop-filter: blur(8px);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.03em;
    }

    /* One centred reading column shared by the conversation and the composer. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottomBlockContainer"] {
        width: 100%;
        max-width: 56rem;
        margin-left: auto;
        margin-right: auto;
        padding-left: clamp(1rem, 3vw, 2rem) !important;
        padding-right: clamp(1rem, 3vw, 2rem) !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3rem !important;
        padding-bottom: 2rem;
        min-width: 0;
    }
    /* The composer floats over a soft fade so messages never look cut off. */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    [data-testid="stBottom"] {
        background: linear-gradient(to top, var(--bg) 55%, rgba(11, 16, 32, 0)) !important;
    }
    [data-testid="stBottomBlockContainer"] { padding-bottom: 1.25rem !important; }

    /* ----- Sidebar ----- */
    [data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--line-soft);
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        white-space: nowrap;
        font: 700 1.5rem/1.25 'Space Grotesk', sans-serif;
        letter-spacing: -0.05em;
    }
    .brand span { color: var(--lime); }
    /* Padding (not margin) so spacing survives Streamlit's markdown wrapper. */
    .brand { padding-bottom: 0.6rem; }
    .sidebar-tagline {
        color: var(--muted);
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        white-space: nowrap;
        margin: 0;
        padding: 0 0 0.9rem;
    }
    .sidebar-group {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: none;
        padding: 1rem 0 0.55rem 0.25rem;
    }
    .sidebar-empty {
        color: var(--muted);
        font-size: 0.85rem;
        margin: 0;
        padding: 0.8rem 0 0 0.25rem;
    }

    /* Tighter rhythm so the history reads as a compact list. */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.2rem; }

    /* Conversation rows: flat, left-aligned, one line with an ellipsis. */
    [data-testid="stSidebar"] .stButton > button {
        min-height: 2.3rem;
        padding: 0.3rem 0.7rem;
        justify-content: flex-start;
        text-align: left;
        overflow: hidden;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #cfd8e6;
        font-size: 0.92rem;
    }
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton > button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--hover);
        border-color: transparent;
        color: var(--text);
    }
    /* Active conversation: raised row with a lime marker. */
    [data-testid="stSidebar"] [class*="st-key-conv_active_"] button {
        background: #17233a;
        color: var(--text);
        box-shadow: inset 3px 0 0 var(--lime);
    }
    /* New conversation: the one accent-coloured action. */
    [data-testid="stSidebar"] [class*="st-key-new_conversation"] button {
        justify-content: center;
        background: rgba(183, 243, 107, 0.09);
        border: 1px solid rgba(183, 243, 107, 0.35);
        color: var(--lime);
        font-weight: 600;
    }
    [data-testid="stSidebar"] [class*="st-key-new_conversation"] button > div,
    [data-testid="stSidebar"] [class*="st-key-new_conversation"] button [data-testid="stMarkdownContainer"] {
        justify-content: center;
        text-align: center;
    }
    [data-testid="stSidebar"] [class*="st-key-new_conversation"] button:hover {
        background: rgba(183, 243, 107, 0.16);
        border-color: var(--lime);
        color: var(--lime);
    }
    [data-testid="stSidebar"] [class*="st-key-new_conversation_active"] button {
        background: rgba(183, 243, 107, 0.16);
        border-color: var(--lime);
    }

    /* Three-dot options button: quiet until the row is hovered or focused. */
    [data-testid="stSidebar"] [data-testid="stPopover"] button {
        min-height: 2.3rem;
        padding: 0.3rem 0;
        justify-content: center;
        border: 1px solid transparent;
        background: transparent;
        color: var(--muted);
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button:hover {
        background: var(--hover);
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] button svg,
    [data-testid="stSidebar"] [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        display: none;
    }
    @media (hover: hover) {
        [data-testid="stSidebar"] [data-testid="stPopover"] {
            opacity: 0;
            transition: opacity 120ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="stPopover"],
        [data-testid="stSidebar"] [data-testid="stPopover"]:focus-within,
        [data-testid="stSidebar"] [data-testid="stPopover"]:has([aria-expanded="true"]) {
            opacity: 1;
        }
    }
    [data-testid="stPopoverBody"] {
        background: var(--raised);
        border: 1px solid var(--line);
    }

    /* ----- Welcome screen ----- */
    .eyebrow {
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .hero {
        text-align: center;
        padding: clamp(1.5rem, 7vh, 4rem) 0 2rem;
    }
    .hero h1 {
        font-size: clamp(2rem, 4.5vw, 3.6rem);
        line-height: 1.08;
        margin: 0 0 1rem;
        overflow-wrap: anywhere;
    }
    .hero h1 span { color: var(--lime); }
    .hero p {
        max-width: 34rem;
        margin: 0 auto;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Suggestion cards are real buttons: the whole card is clickable. */
    [class*="st-key-suggest_"] button {
        width: 100%;
        min-height: 7.5rem;
        height: 100%;
        padding: 1rem 1.1rem;
        align-items: flex-start;
        justify-content: flex-start;
        text-align: left;
        white-space: normal;
        background: linear-gradient(145deg, #152238, #101829);
        border: 1px solid var(--line);
        border-radius: 16px;
        transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
    }
    [class*="st-key-suggest_"] button > div,
    [class*="st-key-suggest_"] button [data-testid="stMarkdownContainer"] {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
    }
    [class*="st-key-suggest_"] button p {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
        margin: 0;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }
    [class*="st-key-suggest_"] button p br { display: none; }
    [class*="st-key-suggest_"] button p strong {
        display: block;
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    [class*="st-key-suggest_"] button:hover {
        transform: translateY(-2px);
        border-color: rgba(183, 243, 107, 0.6);
        background: linear-gradient(145deg, #182740, #121c30);
    }

    /* ----- Conversation ----- */
    .convo-title {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin: 0 0 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0.4rem 0;
        margin-bottom: 0.9rem;
        gap: 0.85rem;
        overflow-wrap: anywhere;
        min-width: 0;
    }
    /* Assistant: open block with a small brand mark. */
    [data-testid="stChatMessageAvatarCustom"] {
        background: #14261f;
        color: var(--lime);
        border: 1px solid #2c4a35;
    }
    /* User: compact bubble aligned to the right, no avatar. */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        width: fit-content;
        max-width: min(85%, 40rem);
        margin-left: auto;
        padding: 0.7rem 1rem;
        background: #17243a;
        border: 1px solid #2a3a55;
        border-radius: 18px 18px 4px 18px;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none; }

    [data-testid="stChatMessage"] h1 { font-size: 1.5rem; }
    [data-testid="stChatMessage"] h2 { font-size: 1.25rem; }
    [data-testid="stChatMessage"] h3 { font-size: 1.08rem; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { line-height: 1.7; }
    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] [data-testid="stTable"],
    [data-testid="stChatMessage"] table {
        max-width: 100%;
        overflow-x: auto;
    }
    [data-testid="stChatMessage"] pre {
        background: #0d1424;
        border: 1px solid var(--line-soft);
        border-radius: 12px;
    }
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td { border-color: var(--line); }

    /* Attachment chips inside a user message. */
    .attachments {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0 0 0.45rem;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        max-width: 100%;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        color: #cfd8e6;
        background: #0f1a2d;
        border: 1px solid var(--line);
        border-radius: 999px;
    }
    .chip .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .chip .detail { color: var(--muted); white-space: nowrap; }
    .chip.ok { border-color: rgba(183, 243, 107, 0.45); }
    .chip.ok .detail { color: var(--lime); }
    .chip.error { border-color: rgba(214, 69, 69, 0.6); }
    .chip.error .detail { color: #ff8a8a; }

    /* Loading indicator while the assistant prepares an answer. */
    .typing { display: inline-flex; align-items: center; gap: 0.5rem; color: var(--muted); font-size: 0.9rem; }
    .typing i {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--lime);
        opacity: 0.35;
        animation: typing-dot 1.2s infinite ease-in-out;
    }
    .typing i:nth-child(2) { animation-delay: 0.15s; }
    .typing i:nth-child(3) { animation-delay: 0.3s; }
    @keyframes typing-dot {
        0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
        40% { opacity: 1; transform: translateY(-3px); }
    }

    /* Errors and warnings share the card look. */
    [data-testid="stAlert"] { border-radius: 12px; }

    /* ----- Composer ----- */
    [data-testid="stChatInput"] {
        background: var(--raised);
        border: 1px solid #33425c;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(183, 243, 107, 0.65);
        box-shadow: 0 0 0 3px rgba(183, 243, 107, 0.12), 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        background: transparent;
        color: var(--text);
    }
    /* Streamlit's inner wrapper draws its own theme-colour focus border; the
       outer composer already provides the lime focus ring. */
    [data-testid="stChatInput"] > div {
        border-color: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #6f7f99; }
    [data-testid="stChatInputSubmitButton"] {
        background: var(--lime);
        color: #0b1020;
        border-radius: 50%;
        transition: transform 120ms ease, background 150ms ease;
    }
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) { transform: scale(1.06); }
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: #1d2a40;
        color: #5d6c86;
    }

    /* ----- Generic buttons and dialog ----- */
    .stButton > button {
        border-radius: 11px;
        border: 1px solid var(--line);
        background: #131e31;
        color: var(--text);
        transition: border-color 150ms ease, color 150ms ease, background 150ms ease;
    }
    .stButton > button:hover {
        border-color: var(--lime);
        color: var(--lime);
        background: #19263b;
    }
    hr { border-color: var(--line); }

    [data-testid="stDialog"] [role="dialog"] {
        background: var(--raised);
        color: var(--text);
        border: 1px solid var(--line);
        border-radius: 16px;
    }
    /* Delete is the destructive action; Cancel stays the neutral default. */
    [data-testid="stDialog"] .stButton > button[kind="primary"] {
        background: #d64545;
        border-color: #d64545;
        color: #ffffff;
    }
    [data-testid="stDialog"] .stButton > button[kind="primary"]:hover {
        background: #e05a5a;
        border-color: #e05a5a;
        color: #ffffff;
    }

    @media (max-width: 700px) {
        /* Keep each history row (title + menu) on one line in the sidebar. */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
        [data-testid="stSidebar"] [data-testid="stColumn"] { min-width: 0 !important; }
        [data-testid="stMainBlockContainer"] { padding-top: 2.5rem !important; }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { max-width: 94%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Text helpers ----------
def escape_md(text):
    """Escape Markdown/LaTeX control characters in user-controlled text."""
    return re.sub(r"([\\`*_{}\[\]()#+!|~$<>])", r"\\\1", text)


def content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)

    return ""


def is_rate_limit_error(error):
    """Recognize Groq/API rate-limit failures without depending on one SDK class."""
    error_text = str(error).lower()
    return (
        type(error).__name__ == "RateLimitError"
        or getattr(error, "status_code", None) == 429
        or "rate_limit_exceeded" in error_text
        or "rate limit reached" in error_text
    )


def rate_limit_retry_seconds(error_text):
    """Parse retry durations such as '6m57.744s' from provider error text."""
    match = re.search(
        r"try again in\s*(?:(\d+(?:\.\d+)?)\s*h)?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*m)?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*s)?",
        error_text,
        flags=re.IGNORECASE,
    )
    if not match or not any(value is not None for value in match.groups()):
        return None
    hours, minutes, seconds = (float(value or 0) for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def user_facing_error_message(error):
    """Return a specific, safe message without exposing raw provider errors."""
    if is_rate_limit_error(error):
        error_text = str(error)
        lowered = error_text.lower()
        if "tokens per day" in lowered or "(tpd)" in lowered:
            limit_name = "daily token limit"
        elif "tokens per minute" in lowered or "(tpm)" in lowered:
            limit_name = "per-minute token limit"
        elif "requests per minute" in lowered or "(rpm)" in lowered:
            limit_name = "request rate limit"
        else:
            limit_name = "current API rate limit"

        retry_seconds = rate_limit_retry_seconds(error_text)
        if retry_seconds is not None:
            if retry_seconds < 60:
                retry_text = f"about {max(1, int(retry_seconds + 0.999))} seconds"
            elif retry_seconds < 3600:
                retry_text = f"about {max(1, int((retry_seconds + 59.999) // 60))} minutes"
            else:
                retry_text = f"about {max(1, int((retry_seconds + 3599.999) // 3600))} hours"
            return (
                f"Queryloom has reached Groq's {limit_name}. "
                f"Please try again in {retry_text}."
            )

        return (
            f"Queryloom has reached Groq's {limit_name}. "
            "Please try again when the limit resets."
        )

    return "Queryloom couldn't complete this response because of a service error. Please try again shortly."


def fallback_title_from_message(message):
    """Create a readable fallback title without using the LLM."""
    title = " ".join(message.split())
    if len(title) > 38:
        title = title[:38].rsplit(" ", 1)[0] + "…"
    return title or DEFAULT_TITLE


def title_from_message(message):
    """Generate a short descriptive title, falling back if the LLM fails."""
    fallback = fallback_title_from_message(message)

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Create a concise, descriptive title for a chat conversation. "
                        "Use 3 to 6 words. Return only the title, without quotes, "
                        "prefixes, explanations, or ending punctuation."
                    )
                ),
                HumanMessage(
                    content=f"Create a title for this first user message:\n{message[:1000]}"
                ),
            ]
        )
        title = " ".join(content_to_text(response.content).split())
        title = title.strip(" \t\n\r\"'`#")
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        if title:
            return title[:50].rstrip()
    except Exception:
        logger.exception("Title generation failed; using fallback title.")

    return fallback


# ---------- Conversation helpers ----------
def create_thread_id():
    return f"thread-{uuid4().hex}"


def load_conversations():
    """Load saved UI transcripts and titles from the PostgreSQL Store."""
    conversations = {}
    try:
        items = store.search(THREAD_NAMESPACE, limit=100)
        for item in items:
            value = item.value
            if not isinstance(value, dict):
                continue
            if not all(key in value for key in ("title", "created_at", "messages")):
                continue
            created_at = value.get("created_at", 0)
            conversations[item.key] = {
                "title": value.get("title", DEFAULT_TITLE),
                "created_at": created_at,
                # Optional field added for recency sorting; older records
                # without it fall back to their creation time.
                "updated_at": value.get("updated_at", created_at),
                "messages": value.get("messages", []),
            }
    except Exception:
        logger.exception("Could not load conversations.")

    return conversations


def save_conversation(thread_id):
    """Persist the title and visible UI transcript for a conversation."""
    conversation = st.session_state.conversations[thread_id]
    try:
        store.put(
            THREAD_NAMESPACE,
            thread_id,
            {
                "title": conversation["title"],
                "created_at": conversation["created_at"],
                "updated_at": conversation.get("updated_at", conversation["created_at"]),
                "messages": conversation["messages"],
            },
        )
    except Exception:
        logger.exception("Could not save conversation metadata.")
        st.warning("Conversation history could not be saved to the database.")


def newest_first(thread_ids):
    conversations = st.session_state.conversations
    return sorted(
        thread_ids,
        key=lambda tid: conversations[tid].get("updated_at", conversations[tid].get("created_at", 0)),
        reverse=True,
    )


def pick_blank_thread():
    """Return an empty conversation to use, creating an in-memory draft if needed.

    Empty drafts are NOT written to the database until the first message is
    sent, so refreshing or pressing "New conversation" never piles up
    duplicate "New chat" records.
    """
    conversations = st.session_state.conversations
    empty = [tid for tid, c in conversations.items() if not c.get("messages")]
    if empty:
        return newest_first(empty)[0]

    thread_id = create_thread_id()
    now = time.time()
    conversations[thread_id] = {
        "title": DEFAULT_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    return thread_id


def delete_conversation(thread_id):
    """Delete one conversation's saved transcript and LangGraph checkpoints.

    Long-term user memory lives under a different Store key and is untouched.
    """
    checkpointer.delete_thread(thread_id)
    store.delete(THREAD_NAMESPACE, thread_id)

    conversations = st.session_state.conversations
    conversations.pop(thread_id, None)

    if st.session_state.active_thread_id == thread_id:
        remaining = newest_first(
            [tid for tid, c in conversations.items() if c.get("messages")]
        )
        st.session_state.active_thread_id = (
            remaining[0] if remaining else pick_blank_thread()
        )


# ---------- Attachments ----------
def index_pdf(uploaded_file):
    """Save an uploaded PDF and index it with the existing ingest_document().

    ingest_document() deletes existing chunks whose `source` equals the file
    name before re-indexing. A short content hash in the stored file name keeps
    two different PDFs that share a name from overwriting each other, while
    re-uploading identical content still replaces its own earlier chunks.
    """
    data = uploaded_file.getvalue()
    stem = re.sub(r"[^\w\- ]", "_", Path(uploaded_file.name).stem).strip() or "document"
    digest = hashlib.sha256(data).hexdigest()[:8]

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{stem}-{digest}.pdf"
    path.write_bytes(data)

    ingest_document(str(path), user_id=USER_ID)
    return path.name


def attachment_chips_html(attachments):
    """Render attachment status chips: status is 'pending', 'ok' or 'error'."""
    chips = []
    for item in attachments:
        status = item.get("status", "ok")
        icon = {"ok": "✓", "error": "!"}.get(status, "…")
        # Older transcripts may still contain a chunk count in `detail`.
        detail = "Ready" if status == "ok" else item.get("detail", "")
        chips.append(
            f'<span class="chip {html.escape(status)}">'
            f'<span class="name">📎 {html.escape(item.get("name", "document"))}</span>'
            f'<span class="detail">{icon} {html.escape(detail)}</span>'
            "</span>"
        )
    return f'<div class="attachments">{"".join(chips)}</div>' if chips else ""


def render_message_body(message):
    """Render one saved message: attachment chips first, then the text."""
    chips = attachment_chips_html(message.get("attachments", []))
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
    if message.get("content"):
        st.markdown(message["content"])


def index_attachments(files, slot, text):
    """Index PDFs one by one, updating the chip row in `slot` as each finishes."""
    attachments = [
        {"name": f.name, "status": "pending", "detail": "Indexing…"} for f in files
    ]

    def paint():
        # The text is shown as plain text inside the same block as the chips.
        slot.markdown(
            attachment_chips_html(attachments)
            + (f"<div>{html.escape(text)}</div>" if text else ""),
            unsafe_allow_html=True,
        )

    paint()
    for position, uploaded_file in enumerate(files):
        try:
            source_name = index_pdf(uploaded_file)

            attachments[position].update(
                status="ok",
                detail="Ready",
                source=source_name,
            )
        except ValueError as error:
            attachments[position].update(status="error", detail=str(error))
        except Exception:
            logger.exception("Indexing %s failed.", uploaded_file.name)
            attachments[position].update(
                status="error", detail="Could not be indexed"
            )
        paint()

    return attachments


# ---------- Callbacks (run before the script reruns) ----------
def on_new_conversation():
    st.session_state.active_thread_id = pick_blank_thread()


def on_select_conversation(thread_id):
    if thread_id in st.session_state.conversations:
        st.session_state.active_thread_id = thread_id


def on_request_delete(thread_id):
    # One-shot request: consumed (popped) in the same run that it is made, so
    # it can never linger and re-open the dialog on a later rerun.
    st.session_state["_delete_request"] = thread_id
    # Remounts the options menus closed so no popover lingers behind the dialog.
    st.session_state.menu_epoch += 1


# ---------- Streaming ----------
TYPING_HTML = '<div class="typing"><i></i><i></i><i></i></div>'


def stream_answer(question, thread_id, placeholder, document_names=None):
    """Stream the answer and return the final assistant text.

    nodes.py emits tokens via get_stream_writer() as {"token": text}, which are
    only visible in the "custom" stream mode. "values" is requested as well so
    the final, authoritative state arrives in the same stream (no extra
    database read, no extra LLM call). Memory-node LLM calls never write
    custom events, so only answer text is displayed.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": USER_ID,
            "active_documents": document_names or [],
        }
    }
    streamed_text = ""
    final_state = None

    for mode, payload in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            if isinstance(payload, dict):
                text = payload.get("token", "")
                if text:
                    streamed_text += text
                    placeholder.markdown(streamed_text + " ▌")
        elif mode == "values":
            final_state = payload

    final_text = streamed_text
    if isinstance(final_state, dict):
        messages = final_state.get("messages", [])
        if messages and messages[-1].type == "ai":
            # Also covers non-streamed replies, e.g. the "message too large" notice.
            final_text = content_to_text(messages[-1].content)

    if not final_text.strip():
        raise RuntimeError("The model returned an empty response.")

    placeholder.markdown(final_text)
    return final_text


# ---------- Initialize app ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.session_state.setdefault("menu_epoch", 0)

if st.session_state.get("active_thread_id") not in st.session_state.conversations:
    st.session_state.active_thread_id = pick_blank_thread()


# ---------- Delete confirmation dialog ----------
@st.dialog("Delete conversation?")
def confirm_delete_dialog(thread_id, title):
    st.markdown(
        f'Are you sure you want to delete "{escape_md(title)}"? '
        "This action cannot be undone."
    )

    cancel_col, delete_col = st.columns(2)

    with cancel_col:
        if st.button("Cancel", key="delete_dialog_cancel", **FULL_WIDTH):
            st.rerun()

    with delete_col:
        delete_clicked = st.button(
            "Delete",
            key="delete_dialog_confirm",
            type="primary",
            **FULL_WIDTH,
        )

    if delete_clicked:
        try:
            delete_conversation(thread_id)
        except Exception:
            logger.exception("Could not delete conversation.")
            st.error("The conversation could not be deleted. Please try again.")
        else:
            st.rerun()


# ---------- Sidebar ----------
active_id = st.session_state.active_thread_id
active_is_blank = not st.session_state.conversations[active_id].get("messages")

with st.sidebar:
    st.markdown(
        '<div class="brand"><span>✳</span>Queryloom</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-tagline">THINK · CREATE · EXPLORE</p>',
        unsafe_allow_html=True,
    )

    # Always first. Highlighted while a blank conversation is open, which is
    # why blank drafts never need their own row in the history list.
    st.button(
        "＋  New conversation",
        key="new_conversation_active" if active_is_blank else "new_conversation",
        on_click=on_new_conversation,
        **FULL_WIDTH,
    )

    # Only conversations that contain messages are listed, newest activity first.
    # Empty drafts are hidden, not deleted.
    history_ids = newest_first(
        [
            tid
            for tid, item in st.session_state.conversations.items()
            if item.get("messages")
        ]
    )

    st.markdown(
        '<div class="sidebar-group">Your conversations</div>',
        unsafe_allow_html=True,
    )

    if not history_ids:
        st.markdown(
            '<p class="sidebar-empty">Your conversations will appear here.</p>',
            unsafe_allow_html=True,
        )

    for saved_thread_id in history_ids:
        saved_conversation = st.session_state.conversations[saved_thread_id]
        is_active = saved_thread_id == active_id

        title = str(saved_conversation.get("title", DEFAULT_TITLE)).strip() or DEFAULT_TITLE
        if len(title) > 40:
            title = title[:39].rstrip() + "…"

        title_col, menu_col = st.columns([5, 1], gap="small", vertical_alignment="center")

        with title_col:
            st.button(
                escape_md(title),
                key=f"{'conv_active_' if is_active else 'conv_'}{saved_thread_id}",
                on_click=on_select_conversation,
                args=(saved_thread_id,),
                **FULL_WIDTH,
            )

        with menu_col:
            with st.popover(
                "⋮",
                help="Conversation options",
                key=f"menu_{saved_thread_id}_{st.session_state.menu_epoch}",
            ):
                st.button(
                    "Delete conversation",
                    key=f"delete_menu_{saved_thread_id}",
                    on_click=on_request_delete,
                    args=(saved_thread_id,),
                    **FULL_WIDTH,
                )


# Open the dialog only in the run where the user explicitly asked for it.
delete_request = st.session_state.pop("_delete_request", None)
if delete_request in st.session_state.conversations:
    confirm_delete_dialog(
        delete_request,
        str(st.session_state.conversations[delete_request].get("title", DEFAULT_TITLE)),
    )


# ---------- Composer input ----------
# Read before the workspace is drawn so the welcome screen can step aside the
# moment something is submitted. The composer itself stays pinned to the bottom.
submission = st.chat_input(
    "Ask Queryloom anything, or attach a PDF…",
    accept_file="multiple",
    file_type=["pdf"],
)
pending_prompt = st.session_state.pop("pending_prompt", None)

if submission:
    typed_text = (submission.get("text") or "").strip()
    attached_files = list(submission.get("files") or [])
else:
    typed_text = ""
    attached_files = []

question = typed_text or (pending_prompt or "").strip()


# ---------- Main workspace ----------
thread_id = st.session_state.active_thread_id
conversation = st.session_state.conversations[thread_id]

if not conversation["messages"] and not (question or attached_files):
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Queryloom · Thinking space</div>
            <h1>Make room for<br><span>better questions.</span></h1>
            <p>
                Explore an idea, understand a difficult concept, or work
                through your next big project. Attach a PDF below to add it
                to your knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        ("Understand a concept", "Explain self-attention with a simple example."),
        ("Explore an idea", "Compare RAG and long-term memory."),
        ("Build something", "Help me plan my next Queryloom feature."),
    ]

    for column, (card_title, card_prompt) in zip(st.columns(3), suggestions):
        with column:
            if st.button(
                f"**{card_title}**  \n{card_prompt}",
                key=f"suggest_{card_title}",
                **FULL_WIDTH,
            ):
                st.session_state.pending_prompt = card_prompt
                st.rerun()
elif conversation["messages"]:
    st.markdown(
        f'<div class="convo-title">{html.escape(str(conversation["title"]))}</div>',
        unsafe_allow_html=True,
    )


# Display saved messages.
for message in conversation["messages"]:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        render_message_body(message)


# ---------- Response ----------
if question or attached_files:
    # A conversation's first real question names it; files alone do not.
    is_first_question = bool(question) and not any(
        m["role"] == "user" and m.get("content") for m in conversation["messages"]
    )

    with st.chat_message("user"):
        if attached_files:
            attachments = index_attachments(attached_files, st.empty(), question)
        else:
            attachments = []
            st.markdown(question)

    user_message = {"role": "user", "content": question}
    if attachments:
        user_message["attachments"] = attachments
    conversation["messages"].append(user_message)
    conversation["updated_at"] = time.time()

    if is_first_question and conversation["title"] in (DEFAULT_TITLE, UPLOAD_TITLE):
        # Instant, free title; upgraded to an LLM title after the first answer.
        conversation["title"] = fallback_title_from_message(question)
    elif not question and conversation["title"] == DEFAULT_TITLE:
        conversation["title"] = UPLOAD_TITLE
    save_conversation(thread_id)

    if not question:
        # Files only: nothing to ask the model, so no LLM call is made.
        indexed = sum(1 for item in attachments if item["status"] == "ok")
        if indexed:
            reply = (
                f"Added {indexed} document{'s' if indexed != 1 else ''} to your "
                "knowledge base. Ask me anything about "
                f"{'them' if indexed != 1 else 'it'}."
            )
        else:
            reply = (
                "None of the attached documents could be indexed. "
                "Please check the files and try again."
            )
        conversation["messages"].append({"role": "assistant", "content": reply})
        save_conversation(thread_id)
        st.rerun()

    answer = None
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        try:
            document_names = list(dict.fromkeys(
            attachment["source"]
            for message in conversation["messages"]
            for attachment in message.get("attachments", [])
            if attachment.get("status") == "ok"
            and attachment.get("source")
            ))
                
            answer = stream_answer(
                question, thread_id, placeholder, document_names
            )
        except Exception as error:
            # Keep the terminal concise; retain the full traceback in the log file.
            error_file_logger.exception("Queryloom response failed")
            if is_rate_limit_error(error):
                logger.error("Groq rate limit exceeded; details saved to %s", ERROR_LOG_PATH)
            else:
                logger.error(
                    "Queryloom response failed (%s); details saved to %s",
                    type(error).__name__,
                    ERROR_LOG_PATH,
                )
            placeholder.error(user_facing_error_message(error))

    if answer is not None:
        # Saved exactly once, and only after a successful response.
        conversation["messages"].append({"role": "assistant", "content": answer})
        conversation["updated_at"] = time.time()

        if is_first_question:
            conversation["title"] = title_from_message(question)

        save_conversation(thread_id)

        # Rerender so the welcome cards disappear and the new title shows.
        st.rerun()
