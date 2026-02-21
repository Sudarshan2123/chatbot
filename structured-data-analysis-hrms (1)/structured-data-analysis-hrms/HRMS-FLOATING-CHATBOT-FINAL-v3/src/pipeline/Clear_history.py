from src.config.configuration import ConfigurationManager
from src.logging import logger
from pymongo import MongoClient
from fastapi import HTTPException
from fastapi.responses import JSONResponse
class ClearHistory:
    def __init__(self):
        try:
            config_obj = ConfigurationManager()
            self.config = config_obj.get_base_config()
            self.client = MongoClient(self.config.MONGODB_URI,self.config.MAX_POOL_SIZE)
        except Exception as e:
            logger.error(f"Error initializing Login class: {e}")
            raise

    def clear_history_process(self,session_id: str):
        try:
            # Connect to the history collection
            history_collection = self.client[self.config.DB_NAME][self.config.HISTORY_COLLECTION_NAME]

            # Count the number of messages for the given session
            message_count = history_collection.count_documents({"SessionId": session_id})
            print(f"Total messages for session {session_id}: {message_count}")

            # Remove all messages for this session
            if message_count > 0:
                result = history_collection.delete_many({"SessionId": session_id})
                print(f"Removed {result.deleted_count} messages for session {session_id}.")

            return {'status': 'success', 'message': 'All chat history cleared'}

        except Exception as e:
            print(f"Error in clearing history for session {session_id}: {e}")
            return {'status': 'error', 'message': str(e)}

    def clear_history(self, session_id: str):
        try:
            result = self.clear_history_process(session_id)
            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")


