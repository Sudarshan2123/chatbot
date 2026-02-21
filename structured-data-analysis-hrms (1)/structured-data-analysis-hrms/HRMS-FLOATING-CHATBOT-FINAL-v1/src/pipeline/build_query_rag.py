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
            # Get the analyzer from connection_data (set in Chat.py)
            analyzer = connection_data.get('analyzer')
            if not analyzer:
                raise ValueError("Analyzer not found in connection_data")
            
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
                # THIS IS THE ONLY CALL NEEDED NOW
                return await self._handle_table_analysis(
                    state_with_intent,
                    analyzer # Pass the analyzer
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
        analyzer: DataAnalyzer # Get analyzer
    ) -> QueryResponse:
        """Handle table analysis workflow - NO DATA LOADING."""
        selected_tables = state_with_intent.get("selected_tables", [])
        
        if not selected_tables:
            return QueryResponse(
                success=False,
                response="",
                error="No tables selected for analysis"
            )
        
        logger.info(f"Analyzing {len(selected_tables)} tables with SQL Agent: {selected_tables}")
        
        

        try:
            # --- START: SIMPLIFIED ANALYSIS ---
            # Perform analysis - this now calls the SQL Agent
            # We run this in a thread to avoid blocking the asyncio loop
            raw_response = await asyncio.to_thread(
                analyzer.analyze_data_with_routing,
                state_with_intent
            )
        
       
            # Extract response text
            if isinstance(raw_response, dict):
                response_text = raw_response.get('output') or raw_response.get('response') or str(raw_response)
            else:
                response_text = str(raw_response) if raw_response else ""
            
            # Clean up response
            pattern_to_strip = r"[\s\n]*(\*Integrated analysis from \d+ tables\*|\*Analysis of [A-Z0-9_]+ \(\d+ rows\)\*)\s*$"
            response_text = re.sub(pattern_to_strip, "", response_text).strip()
            
            # Handle empty response
            if not response_text or len(response_text.strip()) == 0:
                logger.error("Empty response from analyzer")
                return QueryResponse(
                    success=False,
                    response="I apologize, but I couldn't generate a response. Please try rephrasing your question.",
                    selected_tables=selected_tables,
                    error="Empty response from SQL agent"
                )
            
            # Validate response length
            if len(response_text.strip()) < 50:
                logger.warning(f"Short response: {len(response_text)} chars - '{response_text}'")
            
            # Extract analysis type
            analysis_context = state_with_intent.get("analysis_context", {})
            analysis_type = analysis_context.get("analysis_type", "intelligent") if isinstance(analysis_context, dict) else "intelligent"
            
            logger.info(f"Response generated: {len(response_text)} chars")
            
            return QueryResponse(
                success=True,
                response=response_text,
                selected_tables=selected_tables,
                analysis_type=analysis_type,
                metadata={
                    "response_length": len(response_text),
                    "tables_queried": len(selected_tables)
                }
            )
            # --- END: SIMPLIFIED ANALYSIS ---
            
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                response="",
                error=f"Analysis failed: {str(e)}"
            )
    

rag = RAGPipeline()
