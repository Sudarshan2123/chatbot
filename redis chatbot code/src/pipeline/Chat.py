import asyncio
import os
from typing import Any, Dict, Optional
from src.logging import logger
from src.components.token import Token
from src.config.configuration import ConfigurationManager
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import html
from src.components.Chatprocess import Chatbot_Manager
from src.pipeline.build_query_rag import RAGPipeline
from src.pipeline.data_analyzer import DataAnalyzer
from src.pipeline.database_manager import DatabaseManager
from src.pipeline.get_utils import get_gcp_credentials
from src.utils.common import Translate_process_chat

# Global connection data
global default_connection_data
default_connection_data: Optional[Dict[str, Any]] = None


class Chatbot_Pipeline:
    def __init__(self):
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        self.token = Token(self.config)
        credentials = get_gcp_credentials()
        self.Chatbot_manager = Chatbot_Manager(config=self.config, credentials=credentials)
        self.rag_instance = RAGPipeline()
        
        logger.info("Chatbot_Pipeline initialized with dynamic Redis caching")

    async def main_process(self, session_id, user_input) -> dict:
        db_manager = None
        try:
            user_id = session_id
            if not user_id:
                return {'status': 'error', 'message': 'Invalid employee code or password'}
            
            # Connect to database
            db_manager = DatabaseManager()
            if not db_manager.connect():
                return {'status': 'error', 'message': 'Failed to connect to database'}
            
            # Initialize analyzer
            analyzer = DataAnalyzer(config=self.config)
            self.Chatbot_manager.clean_session_history(user_id)
            
            # Get table metadata (fast operation - just metadata, not data)
            tables = db_manager.get_table_names()
            table_metadata = db_manager.get_all_table_metadata()
            
            # Build connection data
            conn_data = {
                'db_manager': db_manager,
                'analyzer': analyzer,
                'table_names': tables,
                'table_metadata': table_metadata,
                'loaded_data': {},
                'schema': db_manager.schema,
                'created_at': asyncio.get_event_loop().time()
            }
            
            global default_connection_data
            default_connection_data = conn_data
            
            # Create agent state
            agent_state = self.config_obj.AgentState(default_connection_data, user_input)
            logger.info(f"Processing query: '{user_input}'")
            
            # STEP 1: Table Router determines which tables are needed
            logger.info("Running table router to determine relevant tables...")
            state_with_intent = analyzer.detect_table_intent(agent_state)
            
            # Check if routing was successful
            if state_with_intent.get('decision') != 'load_selected_tables':
                error_msg = state_with_intent.get('response', 'Table routing failed')
                logger.warning(f"Table routing decision: {state_with_intent.get('decision')}")
                return {'status': 'success', 'answer': error_msg}
            
            # Get dynamically selected tables from router
            selected_tables = state_with_intent.get('selected_tables', [])
            
            if not selected_tables:
                logger.warning("No tables selected by router")
                return {
                    'status': 'success', 
                    'answer': 'I could not determine which tables are relevant for your query. Could you please rephrase?'
                }
            
            logger.info(f"Router selected {len(selected_tables)} table(s): {selected_tables}")
            
            # STEP 2: Load selected tables (Redis first, DB fallback, auto-cache)
            logger.info(f"Loading {len(selected_tables)} table(s)...")
            loaded_data, messages = db_manager.load_multiple_tables(selected_tables)
            
            # Log loading details with cache hit/miss info
            for msg in messages:
                logger.info(msg)
            
            # Check if any tables failed to load
            if not loaded_data:
                logger.error("Failed to load any tables")
                return {
                    'status': 'error', 
                    'answer': 'Unable to load the required data. Please try again.'
                }
            
            # Update state with loaded data
            state_with_intent['loaded_data'] = loaded_data
            default_connection_data['loaded_data'] = loaded_data
            
            # STEP 3: Build RAG response using loaded data
            logger.info("Generating response with RAG pipeline...")
            response = await self.rag_instance.build_query_rag(
                state_with_intent, 
                default_connection_data,
                self.config
            )

            return {'status': 'success', 'answer': response}
            
        except Exception as e:
            logger.error(f"Error in main_process: {e}", exc_info=True)
            response = "Apologies, something went wrong while processing your request. Could you please try again? If the issue persists, feel free to reach out for assistance."
            return {'status': 'success', 'answer': response}
        finally:
            if db_manager:
                db_manager.disconnect()

    async def main_chatbot(self, access_token, input_text, lang):
        """
        Main chatbot endpoint with translation support.
        All table loading is dynamic based on query routing with Redis caching.
        """
        tok_data = self.token.validate_access_token(access_token)
        
        # Translate input if needed
        if lang != "en-US":
            Translated_input = Translate_process_chat(input_text, "en", self.config.API_KEY)
        else:
            Translated_input = input_text
        
        if tok_data:
            session_id = tok_data["session_id"]
            if access_token and Translated_input:
                # Process query (tables loaded dynamically based on routing)
                response = await self.main_process(session_id, Translated_input)
                
                # Extract response text
                respones_text = response.get('answer', '')
                if hasattr(respones_text, 'response'):
                    response_answer = respones_text.response
                else:
                    response_answer = str(respones_text)
                
                # Translate response if needed
                if lang != "en-US":
                    Translated_response = Translate_process_chat(response_answer, lang, self.config.API_KEY)
                else:
                    Translated_response = response_answer
                
                # Update token
                new_access_token = self.token.create_update_token(tok_data)
                encoded_answer = html.escape(str(Translated_response))
                
                # Log conversation
                user_name = self.token.get_user_name_from_access_token(access_token)
                self.Chatbot_manager.mongo_log_chat(user_name, Translated_input, Translated_response)
                
                # Return response with new token
                response = JSONResponse(
                    content={'status': 'success', 'answer': encoded_answer}, 
                    status_code=200
                )
                response.headers['Authorization'] = f"Bearer {new_access_token}"
                return response
            else:
                raise HTTPException(status_code=400, detail="Session ID and input required for chat")
        else:
            raise HTTPException(status_code=400, detail="Invalid Token")
    
    def refresh_cache_for_tables(self, table_names: list):
        """
        Manually refresh Redis cache for specific tables.
        Useful when you know specific tables have been updated.
        
        Args:
            table_names: List of table names to refresh in cache
        """
        try:
            db_manager = DatabaseManager()
            if not db_manager.connect():
                logger.error("Failed to connect to database for cache refresh")
                return False
            
            logger.info(f"Manually refreshing cache for {len(table_names)} tables...")
            results = db_manager.refresh_cache(table_names)
            
            success_count = sum(1 for v in results.values() if v)
            logger.info(f"Cache refresh complete: {success_count}/{len(table_names)} tables refreshed")
            
            db_manager.disconnect()
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
            return False
    
    def clear_all_cache(self):
        """
        Clear all cached tables from Redis.
        Use this when you want to force fresh loads from database.
        """
        try:
            db_manager = DatabaseManager()
            if not db_manager.connect():
                logger.error("Failed to connect to database")
                return False
            
            logger.info("Clearing all cached tables...")
            db_manager.invalidate_cache()
            
            db_manager.disconnect()
            logger.info("All cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get current cache statistics for monitoring.
        Shows which tables are currently cached from previous queries.
        
        Returns:
            Dict with cache stats including cached table names and timestamps
        """
        try:
            db_manager = DatabaseManager()
            if not db_manager.connect():
                return {"error": "Failed to connect to database"}
            
            stats = db_manager.get_cache_stats()
            db_manager.disconnect()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}