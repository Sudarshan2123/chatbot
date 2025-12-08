import logging
# import time  <- Removed, no longer needed
from typing import Dict, List, Optional # <- Removed Generator
import pandas as pd
from langchain_google_vertexai import VertexAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.pipeline.core import (
    AnalysisContext, AnalysisResult, AnalysisType, AnalysisDecision,
    TableMetadata, RoutingDecision, AnalysisConfig, PandasOptionsManager,
    ResponseCleaner, DataAnalysisError, LLMTimeoutError
)
from src.pipeline.response_naturalizer import ResponseNaturalizer
from src.pipeline.table_router import TableRouter
from src.pipeline.safe_analyzer import SafeDataFrameAnalyzer

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Main data analyzer class with proper separation of concerns.
    Orchestrates routing and analysis operations.
    """
    
    def __init__(self, config, enable_naturalization=True):
        """Initialize with optional response naturalization"""
        try:
            from config.Authentication.gcp import load_gcp_credentials
            
            self.config = config
            credentials = load_gcp_credentials()
            
            # Initialize LLM
            self.llm = VertexAI(
                model_name=self.config.RAG_MODEL,
                temperature=0.4,
                max_output_tokens=4096,
                credentials=credentials
            )
            
            # Initialize components
            self.router = TableRouter(self.llm)
            self.analyzer = SafeDataFrameAnalyzer(self.llm)
            
            # NEW: Initialize naturalizer
            self.enable_naturalization = enable_naturalization
            if enable_naturalization:
                self.naturalizer = ResponseNaturalizer(self.llm)
                logger.info("Response naturalization enabled")
            else:
                self.naturalizer = None
            
            logger.info("DataAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataAnalyzer: {e}")
            raise
    
    def detect_table_intent(self, state: dict) -> dict:
        """
        Detect which tables are relevant for the user's query.
        
        Args:
            state: Agent state with user input and available tables
            
        Returns:
            Updated state with routing decision
        """
        user_input = state.get("input", "")
        available_tables = state.get("available_tables", {})
        
        if not available_tables:
            logger.error("No tables available for routing")
            return {
                **state,
                "decision": AnalysisDecision.ERROR_NO_TABLES.value,
                "response": "No database tables are currently loaded."
            }
        
        try:
            # Convert metadata format
            table_metadata = {
                name: TableMetadata(
                    name=name,
                    columns=meta.get("columns", []),
                    data_types=meta.get("data_types", []),
                    row_count=meta.get("row_count", 0)
                )
                for name, meta in available_tables.items()
            }
            
            # Route tables
            routing_decision = self.router.route_tables(user_input, table_metadata)
            
            logger.info(f"Routed to tables: {routing_decision.relevant_tables}")
            
            return {
                **state,
                "selected_tables": routing_decision.relevant_tables,
                "decision": AnalysisDecision.LOAD_SELECTED_TABLES.value,
                "analysis_context": {
                    "analysis_type": routing_decision.analysis_type.value,
                    "relationship_type": routing_decision.relationship_type.value,
                    "confidence": routing_decision.confidence.value,
                    "reasoning": routing_decision.reasoning,
                    "expected_insights": routing_decision.expected_insights
                }
            }
            
        except Exception as e:
            logger.error(f"Error in table routing: {e}", exc_info=True)
            return {
                **state,
                "selected_tables": [],
                "decision": AnalysisDecision.STOP_ANALYSIS.value,
                "analysis_context": f"Unable to determine relevant tables: {str(e)}"
            }
    
    def analyze_data_with_routing(
        self, 
        state: dict
    ) -> str:
        """
        Perform analysis with routing and natural language post-processing.
        
        Args:
            state: Agent state with selected tables and query
            
        Returns:
            Naturalized analysis response string
        """
        selected_tables = state.get("selected_tables", [])
        user_input = state.get("input", "")
        
        if not selected_tables:
            return "No tables selected for analysis."
        
        try:
            # Get loaded data
            all_loaded_data = state.get("loaded_data", {})
            relevant_data = {
                table: all_loaded_data[table] 
                for table in selected_tables 
                if table in all_loaded_data
            }
            
            if not relevant_data:
                return f"Selected tables are not loaded: {', '.join(selected_tables)}"
            
            # Build analysis context
            table_metadata = {
                name: TableMetadata(
                    name=name,
                    columns=list(df.columns),
                    data_types=[str(dtype) for dtype in df.dtypes],
                    row_count=len(df)
                )
                for name, df in relevant_data.items()
            }
            
            context = AnalysisContext(
                user_query=user_input,
                selected_tables=selected_tables,
                routing_decision=None,
                table_metadata=table_metadata
            )
            
            # Perform analysis
            logger.info(f"Analyzing {len(selected_tables)} table(s): {', '.join(selected_tables)}")
            
            raw_response = self.analyzer.analyze_safe(relevant_data, user_input, context)
            
            # NEW: Naturalize the response if enabled
            if self.enable_naturalization and self.naturalizer:
                try:
                    naturalized_response = self.naturalizer.naturalize_response(
                        raw_response,
                        user_input
                    )
                    logger.info("Response naturalized successfully")
                    return naturalized_response
                except Exception as e:
                    logger.warning(f"Naturalization failed, returning raw response: {e}")
                    return raw_response
            
            return raw_response
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            return f"Analysis error: {str(e)}"
    
    def analyze_data_stream(
        self,
        dfs: Dict[str, pd.DataFrame],
        query: str
    ) -> str: # <- Changed return type from Generator to str
        """
        Legacy method for backward compatibility.
        Creates state and calls routing-based analysis.
        Returns a single response string.
        """
        # This would need session state from Streamlit
        # Better to inject dependencies instead
        logger.warning("analyze_data_stream called - prefer routing-based method")
        
        # Create minimal state
        state = {
            "input": query,
            "session_id": "legacy",
            "available_tables": {
                name: {
                    "columns": list(df.columns),
                    "data_types": [str(dt) for dt in df.dtypes],
                    "row_count": len(df)
                }
                for name, df in dfs.items()
            },
            "loaded_data": dfs
        }
        
        # Route tables
        state = self.detect_table_intent(state)
        
        # Check if routing failed
        if state.get("decision") != AnalysisDecision.LOAD_SELECTED_TABLES.value:
            return state.get("response", "Error during table routing.")

        # Analyze
        final_state = self.analyze_data_with_routing(state)
        
        # Return the final response string
        return final_state.get("response", "Analysis completed with no output.")