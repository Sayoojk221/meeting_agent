"""
Orchestrator for managing the meeting agent (llama_index).
1. User can use this orchestrator to run the meeting agent.
2. It will handle the agent workflow and run the meeting agent.
3. In version 1, it will only run similar like production tasks with help of dummy JSON data.
4. In future, it will be extended to handle real-time data and interactions with the meeting agent.
5. User queries will stored in a vector store and used to improve the agent's responses.
6. it have below agents:
    - scheduler
    - notifier
7. it will use localModel called Qwen2.5 to run the agent with help of llama_index and chroma vector store.
"""


from openai import AsyncClient
class Orachestrator:
    def __init__(self, model: str, temparature: float):
        
        self.client = AsyncClient()


    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)