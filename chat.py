from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

from langchain_core.messages import HumanMessage
from app.agent.graph import graph


def create_thread_id():
    return f"thread-{uuid4().hex}"


thread_id = create_thread_id()

print("Queryloom is ready!")
print("Type 'new' to start a new conversation.")
print("Use 'thread <id>' to switch conversations.")
print("Type 'exit', 'bye' or 'stop' to quit.")

while True:
    question = input(f"\nUser:[{thread_id}] ").strip()

    if question.lower() in ["exit", "bye", "stop"]:
        print("Queryloom: Goodbye! See you again.")
        break

    if not question:
        continue

    if question.lower() == "new":
        thread_id = create_thread_id()
        print(f"Started new conversation: {thread_id}")
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
            "thread_id": thread_id,
            "user_id": "local-user",
        }
    }

    try:
        print("\nQueryloom: ", end="", flush=True)

        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "chat_node":
                continue

            content = chunk.content

            if isinstance(content, str):
                print(content, end="", flush=True)

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(block.get("text", ""), end="", flush=True)

        print()

        state = graph.get_state(config).values
        messages = state.get("messages", [])

        print("\n--- Memory Debug ---")
        print("Active messages:", len(messages))
        print("Message roles:", [m.type for m in messages])
        print("User memory:", state.get("summary", "None"))
        print(
            "Conversation summary:",
            state.get("conversation_summary", "Not generated"),
        )

    except Exception as error:
        print(f"\nQueryloom request failed: {error}")
        print("You can try again or type 'new' to start another thread.")