"""
Complete TableRouter Class with ALL methods
Copy this entire class into your data_analyzer.py
"""

import json
import re
import logging
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI

from src.pipeline.core import AnalysisConfig, AnalysisType, ConfidenceLevel, QuerySanitizer, RelationshipType, RoutingDecision, TableMetadata, TableRoutingError

# Assuming these are imported from earlier in the file or from core
# from .core import (
#     RoutingDecision, TableMetadata, AnalysisType, 
#     RelationshipType, ConfidenceLevel, AnalysisConfig,
#     QuerySanitizer, TableRoutingError
# )

logger = logging.getLogger(__name__)


class TableRouter:
    """
    Intelligent table router that uses LLM to determine relevant tables.
    """
    
    def __init__(self, llm: VertexAI):
        """
        Initialize the router with LLM and sanitizer.
        
        Args:
            llm: VertexAI LLM instance
        """
        self.llm = llm
        self.sanitizer = QuerySanitizer()
    
    def route_tables(
        self, 
        user_query: str, 
        available_tables: Dict[str, TableMetadata]
    ) -> RoutingDecision:
        """
        Determine which tables are relevant for the user query.
        
        Args:
            user_query: The user's question
            available_tables: Metadata for all available tables
            
        Returns:
            RoutingDecision with selected tables and context
            
        Raises:
            TableRoutingError: If routing fails
        """
        try:
            # Sanitize query for security
            safe_query = self.sanitizer.sanitize(user_query)
            
            # Check if this is an exploratory query
            if self._is_exploratory_query(safe_query):
                return self._handle_exploratory_query(available_tables)
            
            # Create summary of tables for LLM
            table_summary = self._create_table_summary(available_tables)
            
            # Get routing decision from LLM
            routing_response = self._get_llm_routing(safe_query, table_summary)
            
            # Parse and validate the response
            decision = self._parse_routing_response(routing_response)
            
            if not decision:
                logger.warning("LLM routing failed, using fallback")
                return self._fallback_routing(available_tables)
            
            # Limit to maximum allowed tables
            if len(decision.relevant_tables) > AnalysisConfig.MAX_TABLES_FOR_ROUTING:
                decision.relevant_tables = decision.relevant_tables[:AnalysisConfig.MAX_TABLES_FOR_ROUTING]
            
            logger.info(f"Routed to tables: {decision.relevant_tables}")
            return decision
            
        except Exception as e:
            logger.error(f"Error in table routing: {e}", exc_info=True)
            raise TableRoutingError(f"Failed to route tables: {str(e)}")
    
    def _is_exploratory_query(self, query: str) -> bool:
        """
        Check if query is exploratory in nature (e.g., "show me all data").
        
        Args:
            query: User query string
            
        Returns:
            True if query is exploratory, False otherwise
        """
        exploratory_keywords = [
            'summary', 'overview', 'show me all', 'what data',
            'describe', 'how many tables', 'what tables', 'all tables',
            'table summary', 'what do i have', 'available data', 'data overview'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in exploratory_keywords)
    
    def _handle_exploratory_query(
        self, 
        available_tables: Dict[str, TableMetadata]
    ) -> RoutingDecision:
        """
        Handle exploratory queries by selecting top tables by size.
        
        Args:
            available_tables: All available tables with metadata
            
        Returns:
            RoutingDecision for exploratory analysis
        """
        top_tables = self._select_top_tables(
            available_tables, 
            AnalysisConfig.MAX_TABLES_FOR_ROUTING
        )
        
        return RoutingDecision(
            relevant_tables=top_tables,
            analysis_type=AnalysisType.EXPLORATORY,
            relationship_type=RelationshipType.NONE,
            confidence=ConfidenceLevel.HIGH,
            reasoning="Exploratory query - selected top tables by size",
            expected_insights="General overview of available data"
        )
    
    def _select_top_tables(
        self, 
        available_tables: Dict[str, TableMetadata], 
        max_count: int
    ) -> List[str]:
        """
        Select top N tables by row count.
        
        Args:
            available_tables: Dictionary of table metadata
            max_count: Maximum number of tables to select
            
        Returns:
            List of table names
        """
        sorted_tables = sorted(
            available_tables.items(),
            key=lambda x: x[1].row_count,
            reverse=True
        )
        return [name for name, _ in sorted_tables[:max_count]]
    
    def _create_table_summary(
        self, 
        available_tables: Dict[str, TableMetadata]
    ) -> str:
        """
        Create a readable summary of available tables for the LLM.
        
        Args:
            available_tables: Dictionary of table metadata
            
        Returns:
            Formatted string summary of tables
        """
        summaries = []
        
        for name, metadata in available_tables.items():
            # Get first 10 columns and their types
            columns_preview = metadata.columns[:10]
            dtypes_preview = metadata.data_types[:10]
            
            # Format column information
            column_info = [
                f"{col} ({dtype})" 
                for col, dtype in zip(columns_preview, dtypes_preview)
            ]
            
            # Add ellipsis if there are more columns
            if len(metadata.columns) > 10:
                column_info.append(f"... and {len(metadata.columns) - 10} more")
            
            # Build summary for this table
            summaries.append(
                f"Table: {name}\n"
                f"  Rows: {metadata.row_count:,}\n"
                f"  Columns ({len(metadata.columns)}): {', '.join(column_info)}"
            )
        
        return "\n\n".join(summaries)
    
    def _get_routing_system_prompt(self) -> str:
            """
            Get the system prompt for table routing.
            
            Returns:
                System prompt string for LLM
            """
            return """You are a database table routing expert. Your **only task** is to analyze the user's query and available table information, and output a single, **machine-readable, valid JSON object**.

    CRITICAL RULES:
    1. Select **MAXIMUM 4** tables in total.
    2. The selected tables must be collectively sufficient to answer the user query.
    3. Prioritize tables that have direct relationships (e.g., foreign keys).
    4. The output **MUST NOT** be wrapped in markdown code fences (e.g., ```json...```).

    Respond with **ONLY** valid JSON. Adhere to this **exact structure and format**:

    **{{**
    "relevant_tables": ["TABLE_NAME_1", "TABLE_NAME_2", "..."],
    "analysis_type": "VALUE_A",
    "relationship_type": "VALUE_B",
    "confidence": "VALUE_C",
    "reasoning": "Brief explanation of table selection (max 1 sentence)",
    "expected_insights": "What the resulting analysis should reveal (max 1 sentence)"
    **}}**

    **Possible Values for Fields:**
    * `relevant_tables`: A list of 1 to 4 table names (strings).
    * `analysis_type`: Must be one of: **master_detail** (joining related tables), **aggregation** (calculating summaries), or **simple_lookup** (one or two non-joined tables).
    * `relationship_type`: Must be one of: **foreign_key**, **logical_business**, or **none**.
    * `confidence`: Must be one of: **high**, **medium**, or **low**.

    **FINAL COMMAND: Output ONLY the JSON object, starting with '{{' and ending with '}}'.**"""
    
    def _get_llm_routing(self, query: str, table_summary: str) -> str:
        """
        Get routing decision from LLM.
        
        Args:
            query: User query
            table_summary: Summary of available tables
            
        Returns:
            LLM response string
            
        Raises:
            Exception if LLM call fails
        """
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_routing_system_prompt()),
                ("human", "Available tables:\n{table_info}\n\nUser query: {query}\n\nProvide routing decision:")
            ])
            
            chain = prompt_template | self.llm
            
            response = chain.invoke({
                "query": query,
                "table_info": table_summary
            })
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
            
        except Exception as e:
            logger.error(f"LLM routing invocation failed: {e}")
            raise
    
    def _parse_routing_response(self, response: str) -> Optional[RoutingDecision]:
        """
        Parse LLM routing response using multiple strategies.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            RoutingDecision if parsing successful, None otherwise
        """
        if not response or not response.strip():
            return None
        
        try:
            # Strategy 1: Try direct JSON parse
            try:
                data = json.loads(response.strip())
                return self._validate_and_build_decision(data)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract from markdown code blocks
            patterns = [
                r'```json\s*(\{.*?\})\s*```',
                r'```\s*(\{.*?\})\s*```',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        return self._validate_and_build_decision(data)
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 3: Find JSON-like objects using brace counting
            brace_count = 0
            start_idx = None
            
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        json_str = response[start_idx:i+1]
                        try:
                            data = json.loads(json_str)
                            return self._validate_and_build_decision(data)
                        except json.JSONDecodeError:
                            start_idx = None
                            continue
            
            logger.warning(f"Failed to parse routing response: {response[:200]}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing routing response: {e}")
            return None
    
    def _validate_and_build_decision(self, data: Dict) -> Optional[RoutingDecision]:
        """
        Validate parsed JSON and build RoutingDecision object.
        
        Args:
            data: Parsed JSON dictionary
            
        Returns:
            RoutingDecision if valid, None otherwise
        """
        try:
            # Extract and validate tables list
            tables = data.get('relevant_tables', [])
            if not tables:
                return None
            
            # Ensure it's a list
            if not isinstance(tables, list):
                tables = [str(tables)]
            
            # Clean table names
            tables = [str(t).strip() for t in tables if t]
            if not tables:
                return None
            
            # Build RoutingDecision with proper enum parsing
            return RoutingDecision(
                relevant_tables=tables[:AnalysisConfig.MAX_TABLES_FOR_ROUTING],
                analysis_type=self._parse_enum(
                    data.get('analysis_type', 'intelligent'),
                    AnalysisType,
                    AnalysisType.INTELLIGENT
                ),
                relationship_type=self._parse_enum(
                    data.get('relationship_type', 'none'),
                    RelationshipType,
                    RelationshipType.NONE
                ),
                confidence=self._parse_enum(
                    data.get('confidence', 'medium'),
                    ConfidenceLevel,
                    ConfidenceLevel.MEDIUM
                ),
                reasoning=str(data.get('reasoning', 'No reasoning provided')),
                expected_insights=str(data.get('expected_insights', 'Data analysis'))
            )
            
        except Exception as e:
            logger.error(f"Error validating routing decision: {e}")
            return None
    
    @staticmethod
    def _parse_enum(value: str, enum_class, default):
        """
        Safely parse enum value from string.
        
        Args:
            value: String value to parse
            enum_class: Enum class to parse into
            default: Default value if parsing fails
            
        Returns:
            Enum value or default
        """
        try:
            return enum_class(value.lower())
        except (ValueError, AttributeError):
            return default
    
    def _fallback_routing(
        self, 
        available_tables: Dict[str, TableMetadata]
    ) -> RoutingDecision:
        """
        Fallback routing when LLM routing fails.
        Simply selects top tables by size.
        
        Args:
            available_tables: Dictionary of table metadata
            
        Returns:
            RoutingDecision with fallback selection
        """
        top_tables = self._select_top_tables(
            available_tables, 
            AnalysisConfig.MAX_TABLES_FOR_ROUTING
        )
        
        return RoutingDecision(
            relevant_tables=top_tables,
            analysis_type=AnalysisType.FALLBACK,
            relationship_type=RelationshipType.NONE,
            confidence=ConfidenceLevel.LOW,
            reasoning="Fallback routing - LLM routing unavailable",
            expected_insights="Basic data analysis"
        )