from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from app.agent.state import ChatState
from app.agent.prompts import SYSTEM_PROMPT

load_dotenv()

llm = ChatGroq( model="openai/gpt-oss-20b",temperature=0)

def chat_node(state: ChatState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    response = llm.invoke(messages)

    return {"messages":[response]}