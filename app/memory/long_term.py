from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from app.agent.state import ChatState
from app.memory.short_term import NO_DURABLE_MEMORY

DEFAULT_USER_ID = "local-user"


def get_user_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get(
        "user_id", DEFAULT_USER_ID
    )


def load_long_term_memory_node(
    state: ChatState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    user_id = get_user_id(config)
    namespace = ("queryloom", "users", user_id)

    item = store.get(namespace, "profile")

    if item is None:
        return {}

    summary = item.value.get("summary", "")

    if not isinstance(summary, str) or not summary.strip():
        return {}

    return {"summary": summary.strip()}


def save_long_term_memory_node(
    state: ChatState,
    config: RunnableConfig,
    *,
    store: BaseStore,
) -> dict:
    summary = state.get("summary", "")

    if not isinstance(summary, str) or not summary.strip():
        return {}

    if summary.strip() == NO_DURABLE_MEMORY:
        return {}

    user_id = get_user_id(config)
    namespace = ("queryloom", "users", user_id)

    store.put(
        namespace,
        "profile",
        {"summary": summary.strip()},
    )

    return {}