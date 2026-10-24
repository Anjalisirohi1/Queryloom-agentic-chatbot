from typing import Annotated,TypedDict,NotRequired
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]
    summary: NotRequired[str]
    conversation_summary: NotRequired[str]
