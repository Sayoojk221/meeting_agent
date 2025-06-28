"""
Implementation of the meeting agent orchestrator.

1. User can use this orchestrator to run the meeting agent.
2. It will handle the agent workflow and run the meeting agent.
3. In version 1, it will only run similar like production tasks with help of dummy JSON data.
4. In future, it will be extended to handle real-time data and interactions with the meeting agent.
5. User queries will stored in a vector store and used to improve the agent's responses.
6. it have below agents:
    - scheduler
    - notifier
7. it will use localModel called Qwen2.5 to run the agent with help of llama_index and chroma vector store.
8. Show the agent thinking process in the console.
"""

from agents.scheduler import scheduler_agent
from agents.notifier import notifier_agent
from llama_index.core.agent.workflow import AgentStream, AgentWorkflow
from llama_index.core.memory import Memory
import asyncio

meeting_agent = AgentWorkflow(
    agents=[scheduler_agent, notifier_agent], root_agent="scheduler_agent")

memory = Memory.from_defaults(session_id="meeting_agent", token_limit=40000)

async def process_query(query: str):
    """Process a single query through the meeting agent."""
    handler = meeting_agent.run(query, memory=memory)

    print("\n🤖 Agent Response:")
    print("-" * 50)

    code_block = False

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            content = ev.delta

            if any(skip_phrase in content for skip_phrase in [
                "Action", "Tool",
                "Action Input", "Tool Call", "Input", "handoff", ":" "query_meeting", "schedule_meeting"
            ]):
                continue

            if ("{" in content):
                code_block = True
                continue

            if ("}" in content):
                code_block = False
                continue

            if (code_block):
                continue

            if ("Answer" in content):
                print("\n")

            print(content, end="", flush=True)


async def main():
    """Main interactive chat loop."""
    print("\n🎯 Meeting Agent Chat Interface")
    print("=" * 50)
    print("Welcome! I'm your meeting agent assistant.")
    print("I can help you schedule meetings, send notifications, and more.")
    print("Type 'exit' to quit the chat.\n")

    while True:
        try:
            # Get user input
            print("\n💬 You: ", end="", flush=True)
            user_query = input().strip()

            # Check for exit condition
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye! Thanks for using the meeting agent.")
                break

            # Skip empty queries
            if not user_query:
                print("⚠️  Please enter a query or type 'exit' to quit.")
                continue

            # Process the query
            await process_query(user_query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the meeting agent.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    asyncio.run(main())
