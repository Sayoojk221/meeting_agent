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
from llama_index.core.agent.workflow import AgentStream, AgentWorkflow
import asyncio

meeting_agent = AgentWorkflow(
    agents=[scheduler_agent], root_agent="scheduler_agent")


async def main():

    query = "What are the available meeting slots for this week?"

    handler = meeting_agent.run(
        query
    )

    print("Thinking...")

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):  # showing the thought process
            print(ev.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
