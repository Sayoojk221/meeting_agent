from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama

class Notifier:
    """
    A class that notifies the user about the meeting details.
    """

    def __init__(self):
        self.llm = Ollama(
            base_url="http://localhost:11434",
            model="qwen2.5:14b",
        )

    def send_scheduled_meeting_notification(self, meeting_details: dict):
        """
        Notifies the user about the scheduled meeting details.
        """

        print(f"Sending notification for meeting: {meeting_details}")

        # TODO: Implement the notification logic here

        return "Notification sent successfully"

notifier = Notifier()

system_prompt = f"""
    Identity:
        * Role is help to user with notifying about the meeting details.
        * Notify the user about the scheduled meeting details.
    """

notifier_agent = FunctionAgent(
    name="notifier_agent",
    description="A notifier agent that notifies the user about the meeting details.",
    system_prompt=system_prompt,
    tools=[notifier.send_scheduled_meeting_notification],
    llm=notifier.llm,
)

