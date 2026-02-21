import logging
from typing import Dict, List, Optional
import pandas as pd
from langchain_google_vertexai import ChatVertexAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.pipeline.core import (
    AnalysisContext, AnalysisResult, AnalysisType, AnalysisDecision,
    TableMetadata, RoutingDecision, AnalysisConfig, PandasOptionsManager,
    ResponseCleaner, DataAnalysisError, LLMTimeoutError
)
from src.pipeline.response_naturalizer import ResponseNaturalizer
from src.pipeline.table_router import TableRouter

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Main data analyzer class with proper separation of concerns.
    Orchestrates routing and analysis operations.
    """

    def __init__(self, config, db_manager, enable_naturalization=True):
        """Initialize with optional response naturalization"""
        try:
            from config.Authentication.gcp import load_gcp_credentials

            self.config = config
            self.db_manager = db_manager
            credentials = load_gcp_credentials()

            # Initialize LLM with optimized settings
            self.llm = ChatVertexAI(
                model_name=self.config.RAG_MODEL,
                temperature=0.4,
                max_output_tokens=4096,
                credentials=credentials,
                max_retries=2
            )

            # Initialize components
            self.router = TableRouter(self.llm)

            # Initialize Database Toolkit
            sql_engine = db_manager.get_sqlalchemy_engine()
            db = SQLDatabase(
                engine=sql_engine,
                schema=db_manager.schema,
                include_tables=None
            )

            self.toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            self.tools = self.toolkit.get_tools()

            logger.info(f"Initialized {len(self.tools)} database interaction tools")

            # Initialize naturalizer
            self.enable_naturalization = enable_naturalization
            if enable_naturalization:
                self.naturalizer = ResponseNaturalizer(self.llm)
                logger.info("Response naturalization enabled")
            else:
                self.naturalizer = None

            logger.info("DataAnalyzer (SQL Mode) initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DataAnalyzer: {e}")
            raise

    def execute_sql_securely(self, generated_sql, emp_code):
        """
        Intercepts the LLM's SQL and forces an RBAC filter.
        """
        clean_sql = generated_sql.strip().rstrip(';')
        secure_sql = f"""
            SELECT * FROM ({clean_sql}) AS llm_query
            WHERE employee_code = '{emp_code}'
        """
        return self.db_manager.execute_query(secure_sql)

    async def route_to_tables(self, query: str) -> Dict[str, any]:
        """
        Route query to relevant tables.
        """
        try:
            table_metadata = self.db_manager.get_table_metadata()

            if not table_metadata:
                return {
                    'selected_tables': [],
                    'decision': 'error',
                    'message': 'No tables available'
                }

            tables = [
                TableMetadata(
                    table_name=name,
                    columns=meta.get('columns', []),
                    data_types=meta.get('data_types', []),
                    row_count=meta.get('row_count', 0)
                )
                for name, meta in table_metadata.items()
            ]

            routing_decision = self.router.route(
                query=query,
                available_tables=tables
            )

            return {
                'selected_tables': routing_decision.selected_tables,
                'decision': 'success',
                'routing_decision': routing_decision
            }

        except Exception as e:
            logger.error(f"Error in route_to_tables: {e}")
            return {
                'selected_tables': [],
                'decision': 'error',
                'message': str(e)
            }

    def detect_table_intent(self, state: dict) -> dict:
        """
        Detect which tables are relevant for the user's query.
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
            table_metadata = {
                name: TableMetadata(
                    name=name,
                    columns=meta.get("columns", []),
                    data_types=meta.get("data_types", []),
                    row_count=meta.get("row_count", 0)
                )
                for name, meta in available_tables.items()
            }

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

    def _create_lookup_aware_system_prompt(self, selected_tables: List[str], emp_code: str = None) -> str:
        """
        Create system prompt that enforces lookup table resolution and row-level security.
        """
        lookup_tables = [t for t in selected_tables if 'master' in t.lower() or 'mst' in t.lower()]

        # Define RLS instructions dynamically based on presence of emp_code
        rls_section = ""
        if emp_code and str(emp_code).lower() != "none":
            rls_section = f"""
    0. **ROW-LEVEL SECURITY (MANDATORY)**:
    - Logged-in user employee code: '{emp_code}'
    - EVERY query MUST include: WHERE employee_code = '{emp_code}' OR WHERE emp_code = {emp_code}
    - Check column names in schema - use 'employee_code' or 'emp_code' or 'empid' depending on table
    - If user asks about employee '{emp_code}' or "my" information, this is ALLOWED (it's their own data)
    - If user asks about ANY OTHER employee code or "all employees" or "team members", respond: "You can only access your own employee information."
    - NEVER return data where employee_code != '{emp_code}'
    """
        else:
             # Fallback if emp_code is missing (should not happen with fixes)
            rls_section = """
    0. **ROW-LEVEL SECURITY**:
    - Ensure you only access data explicitly requested by the user.
    """

        base_prompt = f"""You are an expert SQL analyst working with an Oracle/PostgreSQL database.

    **AVAILABLE TABLES**: {', '.join(selected_tables)}

    **CRITICAL RULES**:
    {rls_section}

    1. **ALWAYS USE ALL AVAILABLE TABLES**:
    - You have been provided with {len(selected_tables)} carefully selected tables
    - Do NOT ignore any table, especially lookup/master tables

    2. **LOOKUP TABLE RESOLUTION** (MANDATORY):
    - The following are lookup/master tables: {', '.join(lookup_tables) if lookup_tables else 'None identified'}
    - **ALWAYS JOIN** with lookup tables to resolve IDs to human-readable names
    - NEVER return raw ID values (designation_id, department_id, status_id, etc.)

    3. **SCHEMA INSPECTION**:
    - Call sql_db_schema for ALL tables: {', '.join(selected_tables)}
    - Examine foreign key relationships and column name patterns

    4. **QUERY CONSTRUCTION**:
    - Build comprehensive JOINs that include ALL relevant tables
    - Use LEFT JOIN for optional lookups, INNER JOIN for required data

    5. **VALIDATION**:
    - Before executing, verify you've joined ALL lookup tables

    6. **ERROR HANDLING**:
    - Oracle/Postgres syntax: Do NOT use 'AS' keyword for table aliases
    - Correct: `FROM employee_master emp`

    7. **SECURITY REQUIREMENTS**:
    - NEVER mention table names, column names, or database schema information in response messages
    - If analysis fails, return generic error messages only

    **YOUR TASK**: Answer the user's query using ALL {len(selected_tables)} tables provided,
    resolving ALL ID fields to human-readable values.

    **RESPONSE FORMAT**:
    - Provide COMPLETE information - do NOT truncate your response.
    - **NO TABLES**: If the user asks for "table format", "tabular", or "grid", you must explicitly state: "I cannot provide the output in a table format, but here are the details:"
    - Present the data in a clear LIST or paragraph format.
    - DO NOT use Markdown table syntax (e.g., | Column | Column |).
    - Include ALL relevant columns from the query results.
    """
        return base_prompt

    def _extract_clean_response(self, response: any) -> str:
        """
        Robustly extract clean text from various LangChain response formats.
        """
        try:
            if isinstance(response, str):
                logger.debug("Response is already a string")
                return response.strip()

            if isinstance(response, dict):
                if 'output' in response:
                    return self._extract_clean_response(response['output'])
                if 'result' in response:
                    return self._extract_clean_response(response['result'])
                return str(response)

            if isinstance(response, list):
                text_parts = []
                for idx, item in enumerate(response):
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    return ' '.join(text_parts).strip()
                return str(response)

            if hasattr(response, 'content'):
                return self._extract_clean_response(response.content)

            return str(response)

        except Exception as e:
            logger.error(f"Error in _extract_clean_response: {e}", exc_info=True)
            return str(response)

    def analyze_data_with_routing(self, state: dict) -> str:
        selected_tables = state.get("selected_tables", [])
        user_input = state.get("input", "")
        # FIX: Extract user_id (emp_code) from state
        emp_code = state.get("user_id")

        if not selected_tables:
            return "No tables selected for analysis."

        try:
            sql_engine = self.db_manager.get_sqlalchemy_engine()
            if sql_engine is None:
                return "Analysis error: Database engine is not available."

            db = SQLDatabase(
                engine=sql_engine,
                schema=self.db_manager.schema,
                include_tables=selected_tables
            )

            # FIX: Pass emp_code to prompt creation
            agent_system_prompt = self._create_lookup_aware_system_prompt(selected_tables, emp_code)

            logger.info(f"Creating SQL agent for tables: {selected_tables} (User: {emp_code})")

            sql_agent_executor = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True,
                prefix=agent_system_prompt,
                max_iterations=15,
                max_execution_time=90
            )

            raw_response_dict = sql_agent_executor.invoke({"input": user_input})
            raw_response = self._extract_clean_response(raw_response_dict)

            logger.info(f"Extracted response length: {len(raw_response)} chars")

            return raw_response

        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            return f"Analysis error: {str(e)}"

    # ============================================================================
    # CONTEXT AWARE METHODS
    # ============================================================================

    async def analyze_with_context(
        self,
        query: str,
        tables: List[str],
        conversation_context: List[Dict[str, str]] = None,
        emp_code: str = None  # FIX: Add emp_code argument
    ) -> str:
        """
        Analyze query with conversation context for follow-up questions.
        """
        try:
            if not tables:
                return "No tables selected for analysis."

            sql_engine = self.db_manager.get_sqlalchemy_engine()
            if sql_engine is None:
                return "Analysis error: Database engine is not available."

            db = SQLDatabase(
                engine=sql_engine,
                schema=self.db_manager.schema,
                include_tables=tables
            )

            # FIX: Pass emp_code to context-aware prompt
            agent_system_prompt = self._create_context_aware_system_prompt(
                selected_tables=tables,
                conversation_context=conversation_context,
                emp_code=emp_code
            )

            logger.info(f"Creating SQL agent for tables: {tables} (User: {emp_code})")

            sql_agent_executor = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True,
                prefix=agent_system_prompt,
                max_iterations=15,
                max_execution_time=90
            )

            raw_response_dict = sql_agent_executor.invoke({"input": query})
            raw_response = self._extract_clean_response(raw_response_dict)

            logger.info(f"Extracted response length: {len(raw_response)} chars")

            return raw_response

        except Exception as e:
            logger.error(f"Error in analyze_with_context: {e}", exc_info=True)
            return f"Analysis error: {str(e)}"

    def _create_context_aware_system_prompt(
        self,
        selected_tables: List[str],
        conversation_context: List[Dict[str, str]] = None,
        emp_code: str = None  # FIX: Add emp_code argument
    ) -> str:
        """
        Create system prompt with conversation context and RLS.
        """
        # FIX: Pass emp_code to base prompt creation
        base_prompt = self._create_lookup_aware_system_prompt(selected_tables, emp_code)

        if not conversation_context or len(conversation_context) == 0:
            return base_prompt

        context_lines = []
        for msg in conversation_context[-6:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            if len(content) > 300:
                content = content[:300] + "..."

            if role == 'user':
                context_lines.append(f"Previous User Question: {content}")
            elif role == 'assistant':
                if "emp_code" in content.lower() or "employee" in content.lower():
                    context_lines.append(f"Previous Context: {content[:200]}...")

        if not context_lines:
            return base_prompt

        context_section = "\n".join(context_lines)

        enhanced_prompt = f"""{base_prompt}

    **CONVERSATION CONTEXT** (use this to understand follow-up questions):
    {context_section}

    **IMPORTANT**: If the current query references information from the conversation context
    (like "his", "her", "that employee", etc.), use the context to identify the specific
    employee or entity being discussed."""

        return enhanced_prompt