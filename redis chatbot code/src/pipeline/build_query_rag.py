import asyncio
import logging
import re
from typing import Any, Dict, List, Tuple

from src.entity import QueryResponse
from src.pipeline.data_analyzer import DataAnalyzer, AnalysisDecision

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Refactored RAG pipeline WITHOUT streaming.
    Returns complete responses immediately.
    """
    
    def __init__(self):
        """Initialize RAG pipeline"""
        self.logger = logger
    
    async def build_query_rag(
        self,
        state_with_intent: Dict[str, Any],
        connection_data: Dict[str, Any],
        config: Any
    ) -> QueryResponse:
        """
        Build complete query response using RAG pipeline.
        NO STREAMING - returns complete response.
        
        Args:
            state_with_intent: State with routing decision
            connection_data: Database connection data
            config: Application configuration
            
        Returns:
            QueryResponse with complete analysis results
        """
        try:
            # DataAnalyzer is initialized outside of the try/except if possible, 
            # assuming it doesn't rely on `db_manager` or similar volatile data.
            analyzer = DataAnalyzer(config=config)
            decision = state_with_intent.get("decision")
            
            # Handle general conversation
            if decision == AnalysisDecision.GENERAL_CONVERSATION.value:
                return QueryResponse(
                    success=True,
                    response=state_with_intent.get("response", ""),
                    selected_tables=[],
                    analysis_type="general_conversation"
                )
            
            # Handle table analysis
            elif decision == AnalysisDecision.LOAD_SELECTED_TABLES.value:
                return await self._handle_table_analysis(
                    state_with_intent,
                    connection_data,
                    analyzer
                )
            
            # Handle error cases
            else:
                error_msg = state_with_intent.get(
                    "analysis_context",
                    "Unable to determine relevant tables"
                )
                return QueryResponse(
                    success=False,
                    response="",
                    selected_tables=[],
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                response="",
                error=f"Pipeline error: {str(e)}"
            )
    
    async def _handle_table_analysis(
        self,
        state_with_intent: Dict[str, Any],
        connection_data: Dict[str, Any],
        analyzer: DataAnalyzer
    ) -> QueryResponse:
        """Handle table analysis workflow - returns complete response"""
        selected_tables = state_with_intent.get("selected_tables", [])
        
        if not selected_tables:
            return QueryResponse(
                success=False,
                response="",
                error="No tables selected for analysis"
            )
        
        logger.info(f"Analyzing {len(selected_tables)} tables: {selected_tables}")
        
        # Load tables on demand
        loaded_data, load_messages = await self.load_tables_on_demand(
            selected_tables,
            connection_data
        )
        
        # Update state with loaded data
        state_with_intent["loaded_data"] = {
            table: loaded_data[table]
            for table in selected_tables
            if table in loaded_data
        }
        
        if not state_with_intent["loaded_data"]:
            error_msg = f"Failed to load tables: {selected_tables}"
            logger.error(error_msg)
            return QueryResponse(
                success=False,
                response="",
                error=error_msg
            )
        
        try:
            # Perform analysis - get COMPLETE response (NO STREAMING)
            raw_response = analyzer.analyze_data_with_routing(state_with_intent)
            
            # FIX: Safely extract response text, handling potential dictionary wrapping 
            # from LangChain agent outputs if the analyzer failed to extract it.
            if isinstance(raw_response, dict):
                # Try common keys like 'output' or 'response'
                response_text = raw_response.get('output') or raw_response.get('response')
                if not isinstance(response_text, str):
                    # Fallback to stringifying the whole dictionary if necessary
                    response_text = str(raw_response)
            else:
                # Assume it's a string or can be safely cast
                response_text = str(raw_response)
            
            pattern_to_strip = r"[\s\n]*(\*Integrated analysis from \d+ tables\*|\*Analysis of [A-Z0-9_]+ \(\d+ rows\)\*)\s*$"
            response_text = re.sub(pattern_to_strip, "", response_text).strip()
            # Validate response
            if len(response_text.strip()) < 50: # The line that previously failed
                logger.warning(f"Short response: {len(response_text)} chars")
                response_text += (
                    f"\n\n⚠️ Note: Response length is {len(response_text)} characters. "
                    "If incomplete, there may be data processing issues."
                )
            
            # Extract analysis type safely
            analysis_context = state_with_intent.get("analysis_context", {})
            if isinstance(analysis_context, dict):
                analysis_type = analysis_context.get("analysis_type", "intelligent")
            else:
                analysis_type = "intelligent"
            
            return QueryResponse(
                success=True,
                response=response_text,
                selected_tables=selected_tables,
                analysis_type=analysis_type,
                metadata={
                    "response_length": len(response_text),
                    "tables_loaded": len(state_with_intent["loaded_data"]),
                    "load_messages": load_messages
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                response="",
                error=f"Analysis failed: {str(e)}"
            )
    
    async def load_tables_on_demand(
        self,
        selected_tables: List[str],
        conn_data: Dict[str, Any],
        clear_previous: bool = True
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load selected tables on demand with proper async handling.
        
        Args:
            selected_tables: List of table names to load
            conn_data: Connection data with db_manager
            clear_previous: Whether to clear previously loaded tables
            
        Returns:
            Tuple of (loaded_data dict, messages list)
        """
        db_manager = conn_data.get('db_manager')
        if not db_manager:
            logger.error("No database manager in connection data")
            return {}, ["Error: No database manager available"]
        
        loaded_data = conn_data.get('loaded_data', {})
        messages = []
        
        # Clear previous data if requested
        if clear_previous and loaded_data:
            cleared_count = len(loaded_data)
            cleared_tables = list(loaded_data.keys())
            loaded_data.clear()
            conn_data['loaded_data'] = loaded_data
            
            messages.append(
                f"Cleared {cleared_count} previously loaded tables: {cleared_tables}"
            )
            logger.info(f"Cleared previous data: {cleared_tables}")
        
        # Determine which tables need loading
        tables_to_load = [
            table for table in selected_tables 
            if table not in loaded_data
        ]
        
        if tables_to_load:
            logger.info(f"Loading {len(tables_to_load)} tables: {tables_to_load}")
            
            try:
                # Load tables asynchronously (run sync code in thread)
                # This is non-streaming and returns complete dataframes.
                new_data, load_msgs = await asyncio.to_thread(
                    db_manager.load_multiple_tables,
                    tables_to_load
                )
                
                loaded_data.update(new_data)
                conn_data['loaded_data'] = loaded_data
                messages.extend(load_msgs)
                
                logger.info(f"Successfully loaded {len(new_data)} tables")
                
            except Exception as e:
                error_msg = f"Error loading tables: {str(e)}"
                logger.error(error_msg, exc_info=True)
                messages.append(error_msg)
        else:
            messages.append(
                f"All {len(selected_tables)} selected tables already loaded"
            )
        
        return loaded_data, messages


# Create singleton instance for backward compatibility
rag = RAGPipeline()
