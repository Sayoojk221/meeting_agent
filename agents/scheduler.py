"""
    1. Scheduler agent that manages the scheduling of meetings.
    2. its have below tools:
        - query_meeting
        - schedule_meeting
        - cancel_meeting
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
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.core.node_parser import JSONNodeParser
from llama_index.readers.json import JSONReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.ollama import Ollama
from datetime import datetime

current_date = datetime.now().strftime("%Y-%m-%d")


class Scheduler:
    """
    Scheduler agent that manages the scheduling of meetings.

    Methods:
        schedule_meeting: Schedules a meeting.
        cancel_meeting: Cancels a meeting.
        query_meeting: Queries meeting details, available meeting slots, meeting status, etc.
    """

    def __init__(self):
        self.llm = Ollama(
            base_url="http://localhost:11434",
            model="qwen2.5:14b",
        )

        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5")

        self.db = self._load_db()

        self.query_engine = self.db.as_query_engine(llm=self.llm)

    def _configure_vector_store(self):
        db = chromadb.PersistentClient(path="./db")
        collection = db.get_or_create_collection("meeting_records")

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        return vector_store

    def _load_meeting_records(self):
        reader = JSONReader()

        documents = reader.load_data(input_file="fake-notes/meating-1.json")

        return documents

    def _check_db_exists(self):
        exists = os.path.exists("./db") and os.path.isdir("./db")
        return bool(exists)

    def _meeting_to_node(self, meeting: dict) -> list:

        parser = JSONNodeParser()

        nodes = parser.get_nodes_from_documents(
            documents=[Document(text=json.dumps(meeting))])

        return nodes

    def _load_db(self):
        is_db_exists = self._check_db_exists()

        vector_store = self._configure_vector_store()

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

        db = None

        if is_db_exists:
            db = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, storage_context=storage_context, embed_model=self.embed_model)
        else:
            print("\n Database does not exist. Creating a new one.\n")
            documents = self._load_meeting_records()
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
            db = VectorStoreIndex.from_documents(
                embed_model=self.embed_model,
                storage_context=storage_context,
                documents=documents,
                transformations=[text_splitter],
            )

        return db

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

        meeting = {
            "meeting_id": meeting_id,
            "start_time": meeting_details["start_time"],
            "end_time": meeting_details["end_time"],
            "attendees": meeting_details["attendees"],
            "title": meeting_details["title"],
            "location": meeting_details.get("location", "Online"),
            "meeting_date": meeting_details["meeting_date"],
            "status": "scheduled"
        }

        try:
            nodes = self._meeting_to_node(meeting)

            self.db.insert_nodes(nodes=nodes, embed_model=self.embed_model)

            return f"Meeting scheduled successfully with ID: {meeting_id}"
        except Exception as e:
            print(f"\nError scheduling meeting: {e}\n")
            return "An error occurred while scheduling the meeting. Please try again later."

    def cancel_meeting(self, doc_id: str) -> str:
        # TODO: Implement the cancel meeting logic here
        pass

    def query_meeting(self, query: str) -> str:
        """
        Common query to get meeting details, available meeting slots, meeting status, etc.
        This method will search through all meetings including scheduled and cancelled ones.

        Args:
            query (str): The query to search for meeting details, available meeting slots, meeting status, etc.
                        Can search by meeting ID, title, date, attendees, or status.

        Returns:
            str: The response from the query engine with meeting details and current status.

        """

        try:
            # Add context about cancelled meetings to the query
            enhanced_query = f"""
            {query}
            
            Please provide comprehensive meeting information including:
            - Meeting ID
            - Title and date
            - Current status (scheduled, cancelled, etc.)
            - If cancelled, include cancellation details
            - Attendees and location
            
            If multiple records exist for the same meeting ID, prioritize the most recent status.
            """
            
            result = self.query_engine.query(enhanced_query)
            
            # Add helpful context to the response
            response = str(result)
            if "cancelled" in response.lower():
                response += "\n\nNote: This meeting has been cancelled. If you need to reschedule, please create a new meeting."
            
            return response
            
        except Exception as e:
            print(f"\nError querying meeting: {e}\n")
            return "An error occurred while querying the meeting. Please try again or contact support."


scheduler = Scheduler()

system_prompt = f"""
       Identity:
        * Role is to help users with scheduling meetings.
        * Query meeting details, available meeting slots, meeting status, etc. if the user asks for meeting details.
        * Schedule a meeting if the user provides meeting details.
        * Cancel a meeting if the user requests to cancel a meeting.
        * When scheduling, query available meeting slots using today's date ({current_date}) as the reference date.
        * Always provide helpful guidance to users about meeting IDs and status.
      
       Tools:
        * query_meeting: Query meeting details, available meeting slots, meeting status, etc. Use this to find meetings by ID, title, date, or attendees.
        * schedule_meeting: Schedule a meeting if the user provides meeting details. Returns a meeting ID that can be used for future reference.
        * cancel_meeting: Cancel a meeting by providing the meeting ID. If user doesn't know the meeting ID, use query_meeting first to find it.
        
       Cancellation Process:
        1. If user wants to cancel but doesn't provide meeting ID, first use query_meeting to help them find the meeting.
        2. Once you have the meeting ID, use cancel_meeting with the specific meeting ID.
        3. Always confirm the meeting details before cancellation.
    """

scheduler_agent = ReActAgent(
    name="scheduler_agent",
    description="A meeting scheduler agent that manages the scheduling of meetings.",
    system_prompt=system_prompt,
    tools=[
        scheduler.query_meeting,
        scheduler.schedule_meeting,
        scheduler.cancel_meeting
    ],
    llm=scheduler.llm,
)
