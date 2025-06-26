"""
    1. Scheduler agent that manages the scheduling of meetings.
    2. its have below tools:
        - query_available_meeting
        - schedule_meeting
        - cancel_meeting
        - sync_meeting_records_with_system_db
    3. query available meeting if the user asks for available meeting slots. 
       Once its fetched new meeting slots, it will sync the meeting records with system db.
    4. Convert this class to functions, each function should have detailed description and types
    5 Use ReactAgent from llama_index to run the agent with help of chroma vector store.
"""

import chromadb
import random
import os
import json

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.agent.workflow import ReActAgent 
from llama_index.llms.ollama import Ollama


class Scheduler:
    """
    Scheduler agent that manages the scheduling of meetings.

    Methods:
        sync_meeting_records_with_system_db: Syncs meeting records with the system database.
        schedule_meeting: Schedules a meeting.
        cancel_meeting: Cancels a meeting.
        query_available_meeting: Queries available meeting slots.
    """

    def __init__(self):
        self.llm = Ollama(
            base_url="http://localhost:11434",
            model="qwen2.5:14b",
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")

        self.db = self._sync_meeting_records_with_system_db()

    def _configure_db(self):
        db = chromadb.PersistentClient(path="./db")
        collection = db.get_or_create_collection("meeting_records")

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        return vector_store

    def _load_meeting_records(self):
        """
        Loads meeting records from the fake-notes directory and converts them to nodes.
        Returns:
            list: List of meeting nodes.
        """

        meeting_nodes = []
        for file_path in os.listdir("fake-notes"):
            if file_path.endswith(".json"):
                with open(os.path.join("fake-notes", file_path), "r") as f:
                    meeting = json.load(f)
                    for meeting_record in meeting:
                        node = self._meeting_to_node(meeting_record)
                        meeting_nodes.append(node)

        return meeting_nodes

    def _meeting_to_node(self, meeting):
        # Combine relevant fields into the node text
        attendees = ", ".join(meeting["attendees"])

        text = (
            f"Meeting: {meeting['title']}\n"
            f"Date: {meeting['date']}\n"
            f"Location: {meeting['location']}\n"
            f"Attendees: {attendees}"
        )
        # Create a TextNode with the meeting details

        node = TextNode(
            text=text,
            metadata={
                "meeting_id": meeting["meeting_id"],
                "date": meeting["date"],
                "start_time": meeting["start_time"],
                "end_time": meeting["end_time"],
                "title": meeting["title"],
                "location": meeting["location"],
                "attendees": meeting["attendees"],
                "status": meeting["status"]
            }
        )

        return node

    def _sync_meeting_records_with_system_db(self):
        nodes = self._load_meeting_records()

        vector_store = self._configure_db()

        index = VectorStoreIndex(
            vector_store=vector_store,
            embed_model=self.embed_model,
            nodes=nodes,
            node_parser=SentenceSplitter(
                chunk_size=512, chunk_overlap=20
            )
        )

        return index

    def schedule_meeting(self, meeting_details: dict) -> str:
        """
        Schedules a meeting.
        Args:
            meeting_details (dict): Details of the meeting to be scheduled.
                - Example:
                    start_time (str): Start time of the meeting in ISO format.
                    end_time (str): End time of the meeting in ISO format.
                    attendees (list): List of attendees for the meeting.
                    title (str): Title of the meeting.
                    meeting_date (str): Date of the meeting in ISO format.
        Returns:    
            str: Confirmation message with the meeting ID.
        """

        meeting_id = f'M{random.randint(1000, 9999)}'

        print(f"Scheduling meeting with ID: {meeting_id}")
        print(f"Meeting Details: {meeting_details}")

        meeting = {
            "meeting_id": meeting_id,
            "start_time": meeting_details["start_time"],
            "end_time": meeting_details["end_time"],
            "attendees": meeting_details["attendees"],
            "title": meeting_details["title"],
            "location": meeting_details.get("location", "Online"),
            "date": meeting_details["meeting_date"],
            "status": "scheduled"
        }

        new_meeting_node = self._meeting_to_node(meeting)

        # Add the new meeting node to the vector store
        self.db.insert(new_meeting_node)

        return "Meeting scheduled successfully with ID: " + meeting_id

    def cancel_meeting(self):
        """
        Cancels a meeting.
        """
        # Logic to cancel a meeting
        pass

    def query_meeting_slots(self, query: str) -> str:
        """
        Queries available meeting slots.
        Args:
            query (str): Query string to search for available meeting slots.
        Returns:
            str: Response containing available meeting slots.
        """
        query_engine = self.db.as_query_engine(
            llm=self.llm
        )

        return query_engine.query(query)


scheduler = Scheduler()

scheduler_agent = ReActAgent(
    name="scheduler_agent",
    description="A meeting scheduler agent that manages the scheduling of meetings.", 
    system_prompt="You are a meeting scheduler agent. Your task is to manage the scheduling of meetings. You can query available meeting slots, schedule meetings, cancel meetings. You will use the provided tools to perform these tasks. Make sure to provide detailed responses and follow the user's instructions carefully.",   
    tools=[
        scheduler.query_meeting_slots,
        scheduler.schedule_meeting, scheduler.cancel_meeting],
    llm=scheduler.llm,
)
