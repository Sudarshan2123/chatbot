"""
Module for handling the core Chatbot Pipeline logic, including
intent classification, database interactions, and translation.
"""
import asyncio
import html
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from langchain_google_vertexai import ChatVertexAI


from src.logging import logger
from src.config.configuration import ConfigurationManager
from src.components.Chatprocess import ChatbotManager
from src.components.streaming import StreamingChatbot
from src.pipeline.build_query_rag import RAGPipeline
from src.pipeline.data_analyzer import DataAnalyzer
from src.pipeline.database_manager import DatabaseManager
from src.pipeline.get_utils import get_gcp_credentials
from src.pipeline.conversation_manager import ConversationManager
from src.utils.session_manager import SessionManager
from src.utils.common import Translate_process_chat
from src.pipeline.intent_classifier import (
    IntentClassifier,
    GreetingGenerator,
    OutOfScopeHandler
)


# Global connection data
DEFAULT_CONNECTION_DATA: Optional[Dict[str, Any]] = None

# Simple greetings that don't need translation
SIMPLE_GREETINGS = [
    'hi', 'hii', 'hiii', 'hiiii', 'hello', 'hey', 'bye', 'thanks', 'thank you'
]


class ChatbotPipeline:
    """Main chatbot pipeline handling intent classification and query processing."""

    def __init__(self):
        """Initialize the chatbot pipeline with all necessary components."""
        self.config_obj = ConfigurationManager()
        self.config = self.config_obj.get_base_config()
        credentials = get_gcp_credentials()
        self.chatbot_manager = ChatbotManager(
            config=self.config,
            credentials=credentials
        )
        self.rag_instance = RAGPipeline()

        # Initialize intent classifier
        self.intent_llm = ChatVertexAI(
            model_name=self.config.RAG_MODEL,
            temperature=0.2,  # Lower temp for consistent classification
            max_output_tokens=512,
            credentials=credentials
        )

        self.intent_classifier = IntentClassifier(self.intent_llm)
        self.intent_classifier.clear_cache()
        self.greeting_generator = GreetingGenerator(self.intent_llm)
        self.out_of_scope_handler = OutOfScopeHandler(self.intent_llm)
        self.conversation_manager = ConversationManager(self.config)
        self.session_manager = SessionManager(self.config)

        # Initialize streaming chatbot
        self.streaming_chatbot = StreamingChatbot(self.intent_llm)

        logger.info(
            "ChatbotPipeline initialized with dynamic intent "
            "classification and streaming"
        )

    def _ensure_session_active(self, user_id: str) -> None:
        """Ensure user session is active or start a new one."""
        if not self.session_manager.is_session_active(user_id):
            self.session_manager.start_session(user_id)
        else:
            self.session_manager.extend_session(user_id)

    def _get_conversation_context(self, user_id: str) -> list:
        """Retrieve and log conversation context for user."""
        context = self.conversation_manager.get_conversation_context(user_id)
        logger.info("Retrieved context for %s: %d messages", user_id, len(context))
        if context:
            logger.info("Last context message: %s...", context[-1]['content'][:100])
        return context

    def _classify_and_log_intent(self, user_input: str):
        """Classify user intent and log the results."""
        classification = self.intent_classifier.classify_intent(user_input)
        logger.info(
            "Intent: %s | Confidence: %s | Requires DB: %s",
            classification.primary_intent,
            classification.confidence,
            classification.requires_data_access
        )
        return classification

    def _handle_greeting_intent(
        self,
        user_input: str,
        classification,
        context: list
    ) -> dict:
        """Handle pure greeting intents."""
        contextual_input = self.conversation_manager.build_context_prompt(
            user_input,
            context
        )
        greeting_response = self.greeting_generator.generate_greeting(
            contextual_input,
            classification.greeting_type or "casual"
        )
        logger.info("Generated greeting for: '%s'", user_input)
        return {'status': 'success', 'answer': greeting_response}

    def _handle_out_of_scope_intent(
        self,
        user_input: str,
        context: list
    ) -> dict:
        """Handle out-of-scope requests."""
        contextual_input = self.conversation_manager.build_context_prompt(
            user_input,
            context
        )
        response = self.out_of_scope_handler.handle_out_of_scope(contextual_input)
        logger.info("Out-of-scope request: '%s'", user_input)
        return {'status': 'success', 'answer': response}

    def _handle_unclear_intent(self, user_input: str, context: list) -> dict:
        """Handle unclear input."""
        contextual_input = self.conversation_manager.build_context_prompt(
            user_input,
            context
        )
        response = self.out_of_scope_handler.handle_unclear(contextual_input)
        logger.info("Unclear input: '%s'", user_input)
        return {'status': 'success', 'answer': response}

    def _process_mixed_intent(self, classification, context: list, user_input: str):
        """Process mixed intent (greeting + query)."""
        contextual_input = self.conversation_manager.build_context_prompt(
            user_input,
            context
        )
        greeting_prefix = self.greeting_generator.generate_greeting(
            contextual_input,
            classification.greeting_type or "casual"
        )
        query_to_process = classification.extracted_query or contextual_input
        logger.info("Mixed intent - processing query: '%s'", query_to_process)
        return greeting_prefix, query_to_process

    def _initialize_database_connection(self) -> Optional[DatabaseManager]:
        """Initialize and connect to database manager."""
        db_manager = DatabaseManager()
        if not db_manager.connect():
            return None
        return db_manager

    def _build_connection_data(
        self,
        db_manager: DatabaseManager
    ) -> Dict[str, Any]:
        """Build connection data dictionary with database and analyzer."""
        analyzer = DataAnalyzer(config=self.config, db_manager=db_manager)

        tables = db_manager.get_table_names()
        table_metadata = db_manager.get_all_table_metadata()

        conn_data = {
            'db_manager': db_manager,
            'analyzer': analyzer,
            'table_names': tables,
            'table_metadata': table_metadata,
            'loaded_data': {},
            'schema': db_manager.schema,
            'created_at': asyncio.get_event_loop().time()
        }
        return conn_data

    def _route_tables(
        self,
        agent_state,
        analyzer: DataAnalyzer
    ) -> Optional[dict]:
        """Route query to determine relevant tables."""
        logger.info("Running table router to determine relevant tables...")
        state_with_intent = analyzer.detect_table_intent(agent_state)

        if state_with_intent.get('decision') != 'load_selected_tables':
            error_msg = state_with_intent.get('response', 'Table routing failed')
            logger.warning(
                "Table routing decision: %s",
                state_with_intent.get('decision')
            )
            return None

        selected_tables = state_with_intent.get('selected_tables', [])
        selected_tables = [t.lower() for t in selected_tables]
        state_with_intent['selected_tables'] = selected_tables

        if not selected_tables:
            logger.warning("No tables selected by router")
            return None

        logger.info(
            "Router selected %d table(s): %s",
            len(selected_tables),
            selected_tables
        )
        return state_with_intent

    async def _generate_rag_response(
        self,
        state_with_intent: dict,
        connection_data: dict
    ) -> str:
        """Generate response using RAG pipeline."""
        logger.info("Generating response with RAG pipeline (SQL Agent)...")
        response = await self.rag_instance.build_query_rag(
            state_with_intent,
            connection_data,
            self.config
        )
        return response

    async def main_process(self, user_input: str, user_id: str) -> dict:
        """
        Main processing pipeline for user queries.

        Args:
            user_input: User's query text
            user_id: Unique user identifier

        Returns:
            Dict with status and answer
        """
        try:
            # Ensure session is active
            self._ensure_session_active(user_id)

            # Get conversation context
            context = self._get_conversation_context(user_id)

            # Classify intent
            classification = self._classify_and_log_intent(user_input)

            # Build contextual input
            contextual_input = self.conversation_manager.build_context_prompt(
                user_input,
                context
            )
            logger.info("Built contextual input: '%s'", contextual_input)

            # Handle pure greetings
            if classification.primary_intent == "greeting":
                return self._handle_greeting_intent(
                    user_input,
                    classification,
                    context
                )

            # Handle out-of-scope requests
            if classification.primary_intent == "out_of_scope":
                return self._handle_out_of_scope_intent(user_input, context)

            # Handle unclear input
            if classification.primary_intent == "unclear":
                return self._handle_unclear_intent(user_input, context)

            # Handle mixed intent (greeting + query)
            greeting_prefix = None
            if classification.primary_intent == "mixed":
                greeting_prefix, query_to_process = self._process_mixed_intent(
                    classification,
                    context,
                    user_input
                )
            else:
                # Pure data query - use contextual input
                query_to_process = contextual_input
                logger.info("Processing contextual query: '%s'", query_to_process)

            # Initialize database connection
            db_manager = self._initialize_database_connection()
            if not db_manager:
                return {
                    'status': 'error',
                    'message': 'Failed to connect to database'
                }

            # Build connection data
            conn_data = self._build_connection_data(db_manager)

            global DEFAULT_CONNECTION_DATA  # pylint: disable=global-statement
            DEFAULT_CONNECTION_DATA = conn_data

            # Initialize chatbot manager session
            self.chatbot_manager.clean_session_history(user_id)

            # Create agent state
            agent_state = self.config_obj.AgentState(
                DEFAULT_CONNECTION_DATA,
                query_to_process,
                user_id=user_id  # Pass the emp_code here
            )
            logger.info("Processing query: '%s'", user_input)

            # Validate employee access
            from src.pipeline.core import QuerySanitizer
            if not QuerySanitizer.validate_employee_access(query_to_process, user_id):
                return {
                    'status': 'success',
                    'answer': 'You can only access your own employee information.'
                }

            # Route tables
            state_with_intent = self._route_tables(
                agent_state,
                conn_data['analyzer']
            )

            if state_with_intent is None:
                return {
                    'status': 'success',
                    'answer': (
                        'I could not determine which tables are relevant '
                        'for your query. Could you please rephrase?'
                    )
                }

            # Generate RAG response
            response = await self._generate_rag_response(
                state_with_intent,
                DEFAULT_CONNECTION_DATA
            )

            # Prepend greeting if mixed intent
            if greeting_prefix:
                final_answer = f"{greeting_prefix}\n\n{response}"
            else:
                final_answer = response

            return {'status': 'success', 'answer': final_answer}

        except (ValueError, KeyError, RuntimeError) as err:
            logger.error("Error in main_process: %s", err, exc_info=True)
            response = (
                "Apologies, something went wrong while processing your request. "
                "Could you please try again?"
            )
            return {'status': 'success', 'answer': response}

    def _translate_input(self, input_text: str, lang: str) -> str:
        """Translate input text if needed."""
        if lang != "en-US":
            if input_text.lower().strip() in SIMPLE_GREETINGS:
                return input_text
            return Translate_process_chat(input_text, "en", self.config.API_KEY)
        return input_text

    def _extract_response_text(self, response: dict) -> str:
        """Extract response text from response object."""
        response_text = response.get('answer', '')
        if hasattr(response_text, 'response'):
            logger.info("Response type: %s", type(response_text))
            return response_text.response
        return str(response_text)

    def _translate_response(self, response_answer: str, lang: str) -> str:
        """Translate response if needed."""
        if lang != "en-US":
            return Translate_process_chat(
                response_answer,
                lang,
                self.config.API_KEY
            )
        return response_answer

    def _log_conversation(
        self,
        user_id: str,
        translated_input: str,
        translated_response: str
    ) -> None:
        """Log conversation to MongoDB and Redis."""
        self.chatbot_manager.mongo_log_chat(
            user_id,
            translated_input,
            translated_response
        )
        self.conversation_manager.store_conversation_turn(
            user_id,
            translated_input,
            translated_response
        )
        logger.info(
            "Stored conversation turn for %s: '%s...' -> '%s...'",
            user_id,
            translated_input[:50],
            translated_response[:50]
        )

    async def main_chatbot(
        self,
        input_text: str,
        lang: str,
        user_id: str
    ) -> JSONResponse:
        """
        Main chatbot endpoint with translation support.

        Args:
            input_text: User's input text
            lang: Language code
            user_id: User identifier

        Returns:
            JSONResponse with chatbot answer
        """
        if not input_text:
            raise HTTPException(status_code=400, detail="No input")

        # Translate input if needed
        translated_input = self._translate_input(input_text, lang)

        # Process query
        response = await self.main_process(translated_input, user_id)

        # Extract response text
        response_answer = self._extract_response_text(response)

        # Translate response if needed
        translated_response = self._translate_response(response_answer, lang)

        # Encode answer
        encoded_answer = html.escape(str(translated_response))

        # Log conversation
        self._log_conversation(user_id, translated_input, translated_response)

        # Return response
        return JSONResponse(
            content={'status': 'success', 'answer': encoded_answer},
            status_code=200
        )

    def refresh_cache_for_tables(self, table_names: list) -> bool:
        """
        Manually refresh Redis cache for specific tables.

        Args:
            table_names: List of table names to refresh in cache

        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager()
            if not db_manager.connect():
                logger.error("Failed to connect to database for cache refresh")
                return False

            logger.info(
                "Manually refreshing cache for %d tables...",
                len(table_names)
            )
            results = db_manager.refresh_cache(table_names)

            success_count = sum(1 for v in results.values() if v)
            logger.info(
                "Cache refresh complete: %d/%d tables refreshed",
                success_count,
                len(table_names)
            )

            return True

        except (ValueError, RuntimeError) as err:
            logger.error("Error refreshing cache: %s", err)
            return False

    def clear_all_cache(self) -> bool:
        """
        Clear all cached tables from Redis.

        Returns:
            True if successful, False otherwise
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

        except (ValueError, RuntimeError) as err:
            logger.error("Error clearing cache: %s", err)
            return False

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get current cache statistics for monitoring.

        Returns:
            Dict with cache stats including cached table names and timestamps
        """
        try:
            db_manager = DatabaseManager()
            if not db_manager.connect():
                return {"error": "Failed to connect to database"}

            stats = db_manager.get_cache_stats()
            return stats

        except (ValueError, RuntimeError) as err:
            logger.error("Error getting cache stats: %s", err)
            return {"error": str(err)}

    async def _stream_text_response(self, text: str) -> Any:
        """Stream text response word by word."""
        for word in text.split():
            yield f"{word} "
            await asyncio.sleep(0.05)

    async def generate_content_stream(
        self,
        user_input: str,
        user_id: str,
        lang: str = "en-US",

    ):
        """
        Generate streaming response for user input.

        Args:
            user_input: User's question/input
            lang: Language code for translation
            user_id: User identifier

        Yields:
            str: Streaming chunks of the response
        """
        try:
            # Translate input if needed
            translated_input = self._translate_input(user_input, lang)

            # Get conversation context
            context = self.conversation_manager.get_conversation_context(user_id)

            # Classify intent
            classification = self.intent_classifier.classify_intent(
                translated_input
            )

            # Handle greetings
            if classification.primary_intent == "greeting":
                response = self.greeting_generator.generate_greeting(
                    translated_input,
                    classification.greeting_type or "casual"
                )
                async for chunk in self._stream_text_response(response):
                    yield chunk
                return

            # Handle out-of-scope
            if classification.primary_intent == "out_of_scope":
                response = self.out_of_scope_handler.handle_out_of_scope(
                    translated_input
                )
                async for chunk in self._stream_text_response(response):
                    yield chunk
                return

            # For data queries, get the full response first then stream it
            response_data = await self.main_process(translated_input, user_id)
            response_answer = self._extract_response_text(response_data)

            # Translate response if needed
            final_response = self._translate_response(response_answer, lang)

            # Log the conversation
            self._log_conversation(user_id, translated_input, final_response)

            # Stream the response word by word
            async for chunk in self._stream_text_response(final_response):
                yield chunk

        except (ValueError, RuntimeError) as err:
            logger.error("Error in generate_content_stream: %s", err)
            error_msg = (
                "Sorry, something went wrong while processing your request."
            )
            async for chunk in self._stream_text_response(error_msg):
                yield chunk
