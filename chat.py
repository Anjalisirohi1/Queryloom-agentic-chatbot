from langchain_core.messages import HumanMessage
from app.agent.graph import graph

thread_id = "thread-1"

config = {
    "configurable":{
        "thread_id": "queryloom-chat"
    }
}

print("Queryloom is ready!")
print("Use 'thread <id>' to switch conversations.")
print("Type 'exit','bye' or 'stop' to quit.")

while True:

    question = input(f"\nUser:[{thread_id}] ").strip()

    if question.lower() in ["exit","bye","stop"]:
        print("Queryloom: Goodbye! See you again.")
        break

    if not question:
        continue

    if question.lower() == "thread" or question.lower().startswith("thread "):
        parts = question.split(maxsplit=1)

        if len(parts) == 1 or not parts[1].strip():
            print("Usage: thread <thread_id>")
            continue

        thread_id = parts[1].strip()
        print(f"Switched to {thread_id}")
        continue

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke({"messages":[HumanMessage(content=question)]},
                          config=config
                          )

    print("\nQueryloom:",result["messages"][-1].content)