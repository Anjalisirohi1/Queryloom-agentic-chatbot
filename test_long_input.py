from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.agent.graph import graph


prompt_path = Path("long_prompt.txt")

if not prompt_path.exists():
    raise FileNotFoundError(
        "long_prompt.txt was not found. Run this script from the project root."
    )

prompt = prompt_path.read_text(encoding="utf-8").strip()

if not prompt:
    raise ValueError("long_prompt.txt is empty!")

# A new thread per run prevents old PostgreSQL history from interfering.
# Both calls within this test use the same thread.
thread_id = f"token-trim-test-{uuid4().hex[:8]}"
config = {"configurable": {"thread_id": thread_id}}

print(f"Test thread: {thread_id}")

print("\nGenerating the long tutorial...")

result = graph.invoke(
    {"messages": [HumanMessage(content=prompt)]},
    config=config,
)

answer = result["messages"][-1].content

print("Tutorial response generated.")
print("Response length:", len(answer), "characters")

print("\nTesting token trimming and follow-up continuity...")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Summarize the tutorial into 10 key takeaways."
            )
        ]
    },
    config=config,
)

print("\nFollow-up response:")
print(result["messages"][-1].content)

print("\n--- Saved conversation summary ---")
print(
    result.get(
        "conversation_summary",
        "No conversation summary stored.",
    )
)

print("\n--- Saved durable user memory ---")
print(result.get("summary", "No durable user memory stored."))