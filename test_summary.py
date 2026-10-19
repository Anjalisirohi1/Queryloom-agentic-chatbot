from langchain_core.messages import HumanMessage, AIMessage
from app.memory.short_term import summarize_messages


messages = [
    HumanMessage(content="My project is called Queryloom."),
    AIMessage(content="Got it. Your project is Queryloom."),
    HumanMessage(content="I am building it using LangGraph and PostgreSQL."),
    AIMessage(content="Those are part of your current stack."),
    HumanMessage(content="I prefer simple explanations with examples."),
]

summary = summarize_messages("", messages)

print("\nGenerated summary:")
print(summary)