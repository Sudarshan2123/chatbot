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
        try:
            analyzer = connection_data.get('analyzer')
            if not analyzer:
                raise ValueError("Analyzer not found in connection_data")

            # FIX: Ensure user_id is carried over if it exists in connection_data but not in state
            if "user_id" not in state_with_intent and "user_id" in connection_data:
                state_with_intent["user_id"] = connection_data["user_id"]

            decision = state_with_intent.get("decision")

            if decision == AnalysisDecision.GENERAL_CONVERSATION.value:
                return QueryResponse(
                    success=True,
                    response=state_with_intent.get("response", ""),
                    selected_tables=[],
                    analysis_type="general_conversation"
                )

            elif decision == AnalysisDecision.LOAD_SELECTED_TABLES.value:
                return await self._handle_table_analysis(
                    state_with_intent,
                    analyzer
                )

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
        analyzer: DataAnalyzer
    ) -> QueryResponse:
        selected_tables = state_with_intent.get("selected_tables", [])

        # FIX: Explicitly log the user_id to verify it is being passed to the thread
        emp_code = state_with_intent.get("user_id")
        logger.info(f"Analyzing {len(selected_tables)} tables for User: {emp_code}")

        if not selected_tables:
            return QueryResponse(
                success=False,
                response="",
                error="No tables selected for analysis"
            )

        try:
            # The analyzer.analyze_data_with_routing method uses state.get("user_id")
            # If state_with_intent has the ID, the SQL Agent will now receive it properly.
            raw_response = await asyncio.to_thread(
                analyzer.analyze_data_with_routing,
                state_with_intent
            )

            if isinstance(raw_response, dict):
                response_text = raw_response.get('output') or raw_response.get('response') or str(raw_response)
            else:
                response_text = str(raw_response) if raw_response else ""

            pattern_to_strip = r"[\s\n]*(\*Integrated analysis from \d+ tables\*|\*Analysis of [A-Z0-9_]+ \(\d+ rows\)\*)\s*$"
            response_text = re.sub(pattern_to_strip, "", response_text).strip()

            if not response_text:
                logger.error("Empty response from analyzer")
                return QueryResponse(
                    success=False,
                    response="I apologize, but I couldn't generate a response. Please try rephrasing your question.",
                    selected_tables=selected_tables,
                    error="Empty response from SQL agent"
                )

            analysis_context = state_with_intent.get("analysis_context", {})
            analysis_type = analysis_context.get("analysis_type", "intelligent") if isinstance(analysis_context, dict) else "intelligent"

            return QueryResponse(
                success=True,
                response=response_text,
                selected_tables=selected_tables,
                analysis_type=analysis_type,
                metadata={
                    "response_length": len(response_text),
                    "tables_queried": len(selected_tables),
                    "user_id": emp_code # Useful for debugging
                }
            )

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                response="",
                error=f"Analysis failed: {str(e)}"
            )

rag = RAGPipeline()