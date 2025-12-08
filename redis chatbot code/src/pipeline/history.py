import os
from fastapi import FastAPI, HTTPException, Depends, Query, APIRouter
from typing import Any, List, Optional, Dict, Annotated
from pydantic import BaseModel
from pymongo import MongoClient
import logging
from datetime import datetime

from src.config.configuration import ConfigurationManager

# Define a simple logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed


# Define a simple Config class for MongoDB connection details
class Config:
    def __init__(self):
        # It's good practice to load from environment variables here,
        # with defaults. This makes your application more flexible.
        self.config_obj=ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        self.MONGODB_URI = os.environ.get("MONGODB_URI", self.config.MONGODB_URI)
        self.DB_NAME = os.environ.get("DB_NAME", self.config.DB_NAME)
        self.HISTORY_COLLECTION_NAME = os.environ.get("HISTORY_COLLECTION_NAME", self.config.HISTORY_COLLECTION_Logs)



# Define a Pydantic model for the chat message.
class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] # Changed to datetime for proper parsing


# Define a Pydantic model for the history response
class ChatHistoryResponse(BaseModel):
    user_id: str # Changed from session_id to user_id
    messages: List[ChatMessage]


# Dependency to get the MongoDB client.
class HistoryPage:
    def __init__(self, config: Config):
        self.config = config
        self.client = None

    def get_mongo_client(self) -> MongoClient:
        if self.client is None:
            try:
                self.client = MongoClient(self.config.MONGODB_URI)
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
                raise  # Re-raise the exception to be caught by FastAPI
        return self.client

    def close_mongo_client(self):
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from MongoDB")

    def parse_timestamp_string(timestamp_str: str) -> datetime | None:
        try:
            # Remove ' IST' and parse
            timestamp_str_no_tz = timestamp_str.replace(" IST", "").strip()
            # Define the format string for parsing
            # %d: day, %m: month, %Y: year, %I: hour (12-hour), %M: minute, %p: AM/PM
            return datetime.strptime(timestamp_str_no_tz, "%d/%m/%Y %I:%M %p")
        except ValueError:
            logger.warning(f"Could not parse timestamp string: '{timestamp_str}'")
            return None

    # Function to get chat history from MongoDB
    def get_chat_history_from_mongodb(
        self,
        user_id: str, # Changed from session_id to user_id
        mongo_client: MongoClient,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str,Any]]:
        """
        Fetches a paginated chat history for a given user ID from MongoDB,
        using the new document structure.
        """
        try:
            db = mongo_client[self.config.DB_NAME]
            history_collection = db[self.config.HISTORY_COLLECTION_NAME]

            # --- CRUCIAL CHANGE: Query by 'user' field ---
            history_cursor = (
                history_collection.find({"user": user_id}) # <-- Corrected field name to 'user'
                .sort("timestamp",-1) # Sort by timestamp ascending to get chronological order
                .skip(offset)
                .limit(limit)
            )
            # --- END OF CHANGE ---

            history = list(history_cursor)
            
            # Optional: Add logging to confirm what was found
            if not history:
                logger.info(f"No history found for user: {user_id} in collection: {self.config.HISTORY_COLLECTION_NAME}")
            else:
                logger.debug(f"Fetched {len(history)} records for user: {user_id}")
                # Consider logging first few characters of a record for verification during debug
                # logger.debug(f"First history record: {history[0]}")

            return history
        except Exception as e:
            logger.error(f" No History", exc_info=True)
            raise  # Re-raise to let FastAPI handle the error (as per your original code)
