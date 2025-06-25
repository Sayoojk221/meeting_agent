"""
    1. Scheduler agent that manages the scheduling of meetings.
    2. its have below tools:
        - query_available_meeting
        - schedule_meeting
        - cancel_meeting
        - sync_meeting_records_with_system_db
    3. query available meeting if the user asks for available meeting slots. 
       Once its fetched new meeting slots, it will sync the meeting records with system db.
    4. 
"""

from llama_index.core import SimpleDirectoryReader

class Scheduler:
    def __init__(self):
        self.embedModel = ""
        

    def query_available_meeting(self, *args, **kwargs):
        # Logic to query available meeting slots
        pass

    def schedule_meeting(self, *args, **kwargs):
        # Logic to schedule a meeting
        pass

    def cancel_meeting(self, *args, **kwargs):
        # Logic to cancel a meeting
        pass

    def sync_meeting_records_with_system_db(self, *args, **kwargs):
        reader = SimpleDirectoryReader("fake-notes/meating-1.json")
        print(reader.load_data(),"data")
        pass

    def run(self, *args, **kwargs):
        # Run the agent with the provided arguments
        pass
    
    

# Example usage
if __name__ == "__main__":
    scheduler_agent = Scheduler()
    scheduler_agent.sync_meeting_records_with_system_db()  # This will read the JSON file and print its contents