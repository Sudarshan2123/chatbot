import asyncio
import os
from typing import Any, Dict, Optional
from src.logging import logger
# from src.components.token import Token
from src.config.configuration import ConfigurationManager
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import html
from src.components.Chatprocess import Chatbot_Manager
from src.components.streaming import StreamingChatbot
from src.pipeline.build_query_rag import RAGPipeline
from src.pipeline.data_analyzer import DataAnalyzer
from src.pipeline.database_manager import DatabaseManager
from src.pipeline.get_utils import get_gcp_credentials
from src.pipeline.conversation_manager import ConversationManager
from src.utils.session_manager import SessionManager
from src.utils.common import Translate_process_chat


# Global connection data
global default_connection_data
default_connection_data: Optional[Dict[str, Any]] = None


class Chatbot_Pipeline:
    def __init__(self):
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        # self.token = Token(self.config)
        credentials = get_gcp_credentials()
        self.Chatbot_manager = Chatbot_Manager(config=self.config, credentials=credentials)
        self.rag_instance = RAGPipeline()

        # 🔧 NEW: Initialize intent classifier
        from langchain_google_vertexai import ChatVertexAI
        self.intent_llm = ChatVertexAI(
            model_name=self.config.RAG_MODEL,
            temperature=0.2,  # Lower temp for consistent classification
            max_output_tokens=512,
            credentials=credentials
        )
        from src.pipeline.intent_classifier import (
            IntentClassifier, 
            GreetingGenerator, 
            OutOfScopeHandler
        )
        self.intent_classifier = IntentClassifier(self.intent_llm)
        self.intent_classifier.clear_cache()  # Clear any cached classifications
        self.greeting_generator = GreetingGenerator(self.intent_llm)
        self.out_of_scope_handler = OutOfScopeHandler(self.intent_llm)
        self.conversation_manager = ConversationManager(self.config)
        self.session_manager = SessionManager(self.config)
        
        # Clear conversation context for test user to start fresh - remove this in production
        # from src.utils.session_store import SessionStore
        # session_store = SessionStore(self.config)
        # session_store.clear_conversation_context("101373")
        
        # Initialize streaming chatbot
        self.streaming_chatbot = StreamingChatbot(self.intent_llm)
        
        logger.info("Chatbot_Pipeline initialized with dynamic intent classification and streaming")

    async def main_process(self, user_input, user_id="101373") -> dict:
        db_manager = None
        try:
            # user_id = session_id
            # if not user_id:
                # return {'status': 'error', 'message': 'Invalid employee code or password'}
            
            # Get conversation context for follow-up queries
            # Ensure session is active or start new one
            if not self.session_manager.is_session_active(user_id):
                self.session_manager.start_session(user_id)
            else:
                self.session_manager.extend_session(user_id)
            
            context = self.conversation_manager.get_conversation_context(user_id)
            logger.info(f"Retrieved context for {user_id}: {len(context)} messages")
            if context:
                logger.info(f"Last context message: {context[-1]['content'][:100]}...")
            
            # 🔧 FIX: Classify intent on raw user input first, then add context for data queries only
            classification = self.intent_classifier.classify_intent(user_input)
            
            logger.info(
                f"Intent: {classification.primary_intent} | "
                f"Confidence: {classification.confidence} | "
                f"Requires DB: {classification.requires_data_access}"
            )
            
            # Build contextual input for ALL intents
            contextual_input = self.conversation_manager.build_context_prompt(user_input, context)
            logger.info(f"Built contextual input: '{contextual_input}'")
            
            # Handle pure greetings
            if classification.primary_intent == "greeting":
                greeting_response = self.greeting_generator.generate_greeting(
                    contextual_input,
                    classification.greeting_type or "casual"
                )
                logger.info(f"Generated greeting for: '{user_input}'")
                return {'status': 'success', 'answer': greeting_response}
            
            # Handle out-of-scope requests
            if classification.primary_intent == "out_of_scope":
                response = self.out_of_scope_handler.handle_out_of_scope(contextual_input)
                logger.info(f"Out-of-scope request: '{user_input}'")
                return {'status': 'success', 'answer': response}
            
            # Handle unclear input
            if classification.primary_intent == "unclear":
                response = self.out_of_scope_handler.handle_unclear(contextual_input)
                logger.info(f"Unclear input: '{user_input}'")
                return {'status': 'success', 'answer': response}
            
            # Handle mixed intent (greeting + query)
            if classification.primary_intent == "mixed":
                # Generate greeting first
                greeting_prefix = self.greeting_generator.generate_greeting(
                    contextual_input,
                    classification.greeting_type or "casual"
                )
                
                # Process the data query portion with context
                query_to_process = classification.extracted_query or contextual_input
                logger.info(f"Mixed intent - processing query: '{query_to_process}'")
            else:
                # Pure data query - use contextual input
                greeting_prefix = None
                query_to_process = contextual_input
                logger.info(f"Processing contextual query: '{query_to_process}'")
            
            # Get singleton database manager
            db_manager = DatabaseManager()
            if not db_manager.connect():
                return {'status': 'error', 'message': 'Failed to connect to database'}
            
            # --- START OF CHANGE ---
            # Pass the db_manager to the analyzer
            analyzer = DataAnalyzer(config=self.config, db_manager=db_manager)
            # --- END OF CHANGE ---
            
            # Initialize analyzer
            # analyzer = DataAnalyzer(config=self.config)
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
                'loaded_data': {}, # This will stay empty!
                'schema': db_manager.schema,
                'created_at': asyncio.get_event_loop().time()
            }

            global default_connection_data
            default_connection_data = conn_data
            
            # Create agent state with contextual input for all data queries
            if classification.primary_intent in ["data_query", "mixed"]:
                agent_state = self.config_obj.AgentState(default_connection_data, query_to_process)
            else:
                agent_state = self.config_obj.AgentState(default_connection_data, contextual_input)
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

            # FIX: Ensure all selected table names are lowercase before proceeding
            # The router should return lowercase if metadata is lowercase, but this is a safeguard.
            selected_tables = [t.lower() for t in selected_tables] 
            state_with_intent['selected_tables'] = selected_tables
            
            if not selected_tables:
                logger.warning("No tables selected by router")
                return {
                    'status': 'success', 
                    'answer': 'I could not determine which tables are relevant for your query. Could you please rephrase?'
                }
            
            logger.info(f"Router selected {len(selected_tables)} table(s): {selected_tables}")
            
          

            logger.info("Generating response with RAG pipeline (SQL Agent)...")
            response = await self.rag_instance.build_query_rag(
                state_with_intent, 
                default_connection_data,
                self.config
            )

              # 🔧 NEW: Prepend greeting if mixed intent
            if greeting_prefix:
                final_answer = f"{greeting_prefix}\n\n{response}"
            else:
                final_answer = response

            return {'status': 'success', 'answer': final_answer}
            
        except Exception as e:
            logger.error(f"Error in main_process: {e}", exc_info=True)
            response = "Apologies, something went wrong while processing your request. Could you please try again?"
            return {'status': 'success', 'answer': response}
        finally:
            pass

    async def main_chatbot(self, input_text, lang, user_id="101373"):
        """
        Main chatbot endpoint with translation support.
        All table loading is dynamic based on query routing with Redis caching.
        """
        # tok_data = self.token.validate_access_token(access_token)
        
        # Translate input if needed
        if lang != "en-US":
            # Skip translation for simple greetings that are universal
            simple_greetings = ['hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you']
            if input_text.lower().strip() in simple_greetings:
                Translated_input = input_text  # Keep original
            else:
                Translated_input = Translate_process_chat(input_text, "en", self.config.API_KEY)
        else:
            Translated_input = input_text
        
        # session_id = tok_data["session_id"]
        if Translated_input:
                # Process query (tables loaded dynamically based on routing)
                response = await self.main_process(Translated_input, user_id)
                    
                respones_text = response.get('answer', '')
                if hasattr(respones_text, 'response'):
                    logger.info(f"Response type: {type(respones_text)}")
                    response_answer = respones_text.response
                else:
                    response_answer = str(respones_text)
                
                # Translate response if needed
                if lang != "en-US":
                    Translated_response = Translate_process_chat(response_answer, lang, self.config.API_KEY)
                else:
                    Translated_response = response_answer
                
                # Update token
                # new_access_token = self.token.create_update_token(tok_data)
                encoded_answer = html.escape(str(Translated_response))
                
                # Log conversation
                # user_name = self.token.get_user_name_from_access_token(access_token)
                user_name = user_id  # Use the passed user_id
                self.Chatbot_manager.mongo_log_chat(user_name, Translated_input, Translated_response)
                
                # Store in Redis for fast context retrieval
                self.conversation_manager.store_conversation_turn(user_name, Translated_input, Translated_response)
                logger.info(f"Stored conversation turn for {user_name}: '{Translated_input[:50]}...' -> '{Translated_response[:50]}...'")
                
                # Return response with new token
                response = JSONResponse(
                    content={'status': 'success', 'answer': encoded_answer}, 
                    status_code=200
                )
                # response.headers['Authorization'] = f"Bearer {new_access_token}"
                return response
            # else:
            #     raise HTTPException(status_code=400, detail="Session ID and input required for chat")
        else:
            raise HTTPException(status_code=400, detail="No input")
    
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
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def generate_content_stream(self, user_input: str, lang: str = "en-US", user_id: str = "101373"):
        """
        Generate streaming response for user input
        
        Args:
            user_input: User's question/input
            lang: Language code for translation
            
        Yields:
            str: Streaming chunks of the response
        """
        try:
            # Translate input if needed
            if lang != "en-US":
                # Skip translation for simple greetings that are universal
                simple_greetings = ['hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you']
                if user_input.lower().strip() in simple_greetings:
                    translated_input = user_input  # Keep original
                else:
                    translated_input = Translate_process_chat(user_input, "en", self.config.API_KEY)
            else:
                translated_input = user_input
            
            # Get conversation context
            context = self.conversation_manager.get_conversation_context(user_id)
            
            # Classify intent on raw input to avoid context pollution
            classification = self.intent_classifier.classify_intent(translated_input)
            
            # Handle different intents
            if classification.primary_intent == "greeting":
                response = self.greeting_generator.generate_greeting(
                    translated_input,
                    classification.greeting_type or "casual"
                )
                # Stream the greeting response
                for word in response.split():
                    yield f"{word} "
                    await asyncio.sleep(0.05)
                return
            
            if classification.primary_intent == "out_of_scope":
                response = self.out_of_scope_handler.handle_out_of_scope(translated_input)
                for word in response.split():
                    yield f"{word} "
                    await asyncio.sleep(0.05)
                return
            
            # For data queries, get the full response first then stream it
            response_data = await self.main_process(translated_input)
            response_text = response_data.get('answer', '')
            
            if hasattr(response_text, 'response'):
                response_answer = response_text.response
            else:
                response_answer = str(response_text)
            
            # Translate response if needed
            if lang != "en-US":
                final_response = Translate_process_chat(response_answer, lang, self.config.API_KEY)
            else:
                final_response = response_answer
            
            # Log the conversation
            self.Chatbot_manager.mongo_log_chat(user_id, translated_input, final_response)
            self.conversation_manager.store_conversation_turn(user_id, translated_input, final_response)
            
            # Stream the response word by word
            words = final_response.split()
            for word in words:
                yield f"{word} "
                await asyncio.sleep(0.05)  # Adjust delay as needed
                
        except Exception as e:
            logger.error(f"Error in generate_content_stream: {e}")
            error_msg = "Sorry, something went wrong while processing your request."
            for word in error_msg.split():
                yield f"{word} "
                await asyncio.sleep(0.05)