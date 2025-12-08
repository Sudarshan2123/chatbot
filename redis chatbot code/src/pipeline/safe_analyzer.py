# import logging
# import time
# from typing import Dict, Generator, List, Optional
# import pandas as pd
# from langchain_google_vertexai import VertexAI
# from langchain.agents import AgentType
# from langchain_experimental.agents import create_pandas_dataframe_agent

# from src.pipeline.core import (
#     AnalysisContext, AnalysisResult, AnalysisType, AnalysisDecision,
#     TableMetadata, RoutingDecision, AnalysisConfig, PandasOptionsManager,
#     ResponseCleaner, DataAnalysisError, LLMTimeoutError
# )
# from src.pipeline.table_router import TableRouter

# logger = logging.getLogger(__name__)

# class SafeDataFrameAnalyzer:
#     """
#     Analyzes DataFrames with safety controls.
#     NO dangerous code execution - uses controlled pandas operations.
#     """
    
#     def __init__(self, llm: VertexAI):
#         self.llm = llm
#         self.response_cleaner = ResponseCleaner()
    
#     def analyze_safe(
#         self,
#         data: Dict[str, pd.DataFrame],
#         query: str,
#         context: AnalysisContext
#     ) -> str:
#         """
#         Analyze data safely without dangerous code execution.
        
#         Args:
#             data: Dictionary of DataFrames to analyze
#             query: User query
#             context: Analysis context
            
#         Returns:
#             Analysis response string
#         """
#         try:
#             if len(data) == 1:
#                 return self._analyze_single_table(data, query)
#             else:
#                 return self._analyze_multiple_tables(data, query, context)
                
#         except Exception as e:
#             logger.error(f"Error in safe analysis: {e}", exc_info=True)
#             raise DataAnalysisError(f"Analysis failed: {str(e)}")
    
#     def _analyze_single_table(
#         self, 
#         data: Dict[str, pd.DataFrame], 
#         query: str
#     ) -> str:
#         """Analyze single table with safety controls"""
#         table_name, df = next(iter(data.items()))
        
#         with PandasOptionsManager():
#             try:
#                 # Use controlled pandas agent with strict limits
#                 agent = create_pandas_dataframe_agent(
#                     self.llm,
#                     df,
#                     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                     verbose=False,  # Reduce noise
#                     # allow_dangerous_code=True,
#                     # handle_parsing_errors=True,
#                     max_iterations=AnalysisConfig.MAX_LLM_ITERATIONS,
#                     early_stopping_method="generate",
#                     prefix=self._get_single_table_prefix(table_name, df)
#                 )
                
#                 response = agent.invoke(query)
#                 output = self._extract_output(response)
#                 cleaned = self.response_cleaner.clean(output, [table_name])
                
#                 return f"{cleaned}\n\n*Analysis of {table_name} ({df.shape[0]:,} rows)*"
                
#             except Exception as e:
#                 logger.error(f"Single table analysis error: {e}")
#                 return self._fallback_single_table_analysis(df, table_name, query)
    
#     def _analyze_multiple_tables(
#         self,
#         data: Dict[str, pd.DataFrame],
#         query: str,
#         context: AnalysisContext
#     ) -> str:
#         """Analyze multiple tables with relationship awareness"""
#         table_names = list(data.keys())
        
#         # Try to merge tables if they have common columns
#         merged_df = self._attempt_smart_merge(data)
        
#         if merged_df is not None and not merged_df.empty:
#             return self._analyze_merged_data(merged_df, query, table_names)
#         else:
#             # Fallback: analyze separately and combine
#             return self._analyze_separately_and_combine(data, query)
    
#     def _attempt_smart_merge(
#         self, 
#         data: Dict[str, pd.DataFrame]
#     ) -> Optional[pd.DataFrame]:
#         """
#         Attempt to merge tables intelligently.
#         Returns None if merge is not feasible.
#         """
#         try:
#             table_names = list(data.keys())
#             if len(table_names) < 2:
#                 return None
            
#             # Start with first table
#             result = data[table_names[0]].copy()
            
#             for i in range(1, len(table_names)):
#                 next_table = data[table_names[i]]
                
#                 # Find common columns
#                 common_cols = list(set(result.columns) & set(next_table.columns))
                
#                 if not common_cols:
#                     logger.warning(f"No common columns for merge with {table_names[i]}")
#                     return None
                
#                 # Ensure types match for merge
#                 for col in common_cols:
#                     result[col] = result[col].astype(str)
#                     next_table[col] = next_table[col].astype(str)
                
#                 # Perform inner join
#                 result = pd.merge(result, next_table, on=common_cols, how='left')
                
#                 # Check if merge resulted in empty DataFrame
#                 if result.empty:
#                     logger.warning(f"Merge resulted in empty DataFrame")
#                     return None
                
#                 logger.info(f"Merged with {table_names[i]}: {result.shape}")
            
#             return result.reset_index(drop=True)
            
#         except Exception as e:
#             logger.error(f"Merge failed: {e}")
#             return None
    
#     def _analyze_merged_data(
#         self,
#         merged_df: pd.DataFrame,
#         query: str,
#         original_tables: List[str]
#     ) -> str:
#         """Analyze merged DataFrame"""
#         with PandasOptionsManager():
#             try:
#                 agent = create_pandas_dataframe_agent(
#                     self.llm,
#                     merged_df,
#                     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                     verbose=False,
#                     # allow_dangerous_code=True,
#                     # handle_parsing_errors=True,
#                     max_iterations=AnalysisConfig.MAX_LLM_ITERATIONS,
#                     early_stopping_method="generate",
#                     prefix=self._get_merged_table_prefix(merged_df, original_tables)
#                 )
                
#                 response = agent.invoke(query)
#                 output = self._extract_output(response)
#                 cleaned = self.response_cleaner.clean(output, original_tables)
                
#                 return f"{cleaned}\n\n*Integrated analysis from {len(original_tables)} tables*"
                
#             except Exception as e:
#                 logger.error(f"Merged analysis error: {e}")
#                 # Fallback to statistics-based response
#                 return self._fallback_merged_analysis(merged_df, query, original_tables)
    
#     def _analyze_separately_and_combine(
#         self,
#         data: Dict[str, pd.DataFrame],
#         query: str
#     ) -> str:
#         """Analyze tables separately and combine insights"""
#         insights = []
        
#         for table_name, df in data.items():
#             try:
#                 result = self._analyze_single_table({table_name: df}, query)
#                 insights.append(result)
#             except Exception as e:
#                 logger.error(f"Error analyzing {table_name}: {e}")
#                 insights.append(f"*Unable to analyze {table_name}*")
        
#         combined = "\n\n---\n\n".join(insights)
#         return f"{combined}\n\n*Combined analysis from {len(data)} tables*"
    
#     def _get_single_table_prefix(self, table_name: str, df: pd.DataFrame) -> str:
#         """Generate prefix for single table analysis"""
#         return f"""You are analyzing the '{table_name}' table with {df.shape[0]:,} rows and {df.shape[1]} columns.

# Columns: {list(df.columns)}

# Provide a clear, comprehensive answer to the user's query based on this data.
# Include specific examples and statistics when relevant.

# IMPORTANT: Your response should be in the 'Final Answer:' section."""
    
#     def _get_merged_table_prefix(
#         self, 
#         merged_df: pd.DataFrame, 
#         original_tables: List[str]
#     ) -> str:
#         """Generate prefix for merged table analysis"""
#         return f"""You are analyzing integrated data from {len(original_tables)} related tables.

# Merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns
# Columns: {list(merged_df.columns)}

# This is a unified dataset - provide ONE comprehensive analysis.
# Do NOT reference individual source tables.
# Focus on answering the user's query with insights from the complete dataset.

# IMPORTANT: Your response should be in the 'Final Answer:' section."""
    
#     def _extract_output(self, response) -> str:
#         """Extract text from various response formats"""
#         if isinstance(response, dict) and 'output' in response:
#             return response['output']
#         elif hasattr(response, 'content'):
#             return response.content
#         return str(response)
    
#     def _fallback_single_table_analysis(
#         self,
#         df: pd.DataFrame,
#         table_name: str,
#         query: str
#     ) -> str:
#         """Fallback analysis using basic statistics"""
#         try:
#             summary = f"**Analysis of {table_name}**\n\n"
#             summary += f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
            
#             # Show column info
#             summary += "**Columns:**\n"
#             for col in df.columns[:10]:
#                 dtype = df[col].dtype
#                 non_null = df[col].notna().sum()
#                 summary += f"- {col} ({dtype}): {non_null:,} values\n"
            
#             # Show basic stats for numeric columns
#             numeric_cols = df.select_dtypes(include=['number']).columns[:5]
#             if len(numeric_cols) > 0:
#                 summary += f"\n**Sample Statistics:**\n"
#                 for col in numeric_cols:
#                     summary += f"- {col}: mean={df[col].mean():.2f}, "
#                     summary += f"min={df[col].min():.2f}, max={df[col].max():.2f}\n"
            
#             return summary
            
#         except Exception as e:
#             logger.error(f"Fallback analysis failed: {e}")
#             return f"Unable to analyze {table_name} due to processing error."
    
#     def _fallback_merged_analysis(
#         self,
#         merged_df: pd.DataFrame,
#         query: str,
#         original_tables: List[str]
#     ) -> str:
#         """Fallback for merged analysis"""
#         try:
#             summary = f"**Integrated Analysis**\n\n"
#             summary += f"Combined dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns\n"
#             summary += f"Source tables: {', '.join(original_tables)}\n\n"
            
#             summary += "The integrated dataset provides comprehensive information "
#             summary += "combining multiple data sources for thorough analysis.\n"
            
#             return summary
            
#         except Exception as e:
#             return "Unable to complete integrated analysis."




import logging
import time
import io
import contextlib
import traceback
from typing import Dict, Generator, List, Optional
import pandas as pd
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.pipeline.core import (
    AnalysisContext, AnalysisResult, AnalysisType, AnalysisDecision,
    TableMetadata, RoutingDecision, AnalysisConfig, PandasOptionsManager,
    ResponseCleaner, DataAnalysisError, LLMTimeoutError
)
from src.pipeline.table_router import TableRouter

logger = logging.getLogger(__name__)


class SafeDataFrameAnalyzer:
    """
    Analyzes DataFrames with safety controls.
    NO dangerous code execution - uses controlled pandas operations.
    """
    
    def __init__(self, llm: VertexAI):
        self.llm = llm
        self.response_cleaner = ResponseCleaner()
    
    def analyze_safe(
        self,
        data: Dict[str, pd.DataFrame],
        query: str,
        context: AnalysisContext
    ) -> str:
        """
        Analyze data safely without dangerous code execution.
        
        Args:
            data: Dictionary of DataFrames to analyze
            query: User query
            context: Analysis context
            
        Returns:
            Analysis response string
            
        Raises:
            DataAnalysisError: If analysis fails
        """
        try:
            if len(data) == 1:
                return self._analyze_single_table(data, query)
            else:
                return self._analyze_multiple_tables(data, query, context)
                
        except Exception as e:
            logger.error(f"Error in safe analysis: {e}", exc_info=True)
            raise DataAnalysisError(f"Analysis failed: {str(e)}")
    
    def _analyze_single_table(
        self, 
        data: Dict[str, pd.DataFrame], 
        query: str
    ) -> str:
        """Analyze single table using controlled code execution."""
        table_name, df = next(iter(data.items()))
        
        with PandasOptionsManager():
            try:
                prompt = self._get_single_table_analysis_prompt(table_name, df, query)
                response = self.llm.invoke(prompt)
                llm_output = self._extract_output(response)
                cleaned = self.response_cleaner.clean(llm_output, [table_name])
                
                return f"{cleaned}\n\n*Analysis of {table_name} ({df.shape[0]:,} rows)*"
                
            except Exception as e:
                logger.error(f"Single table analysis error: {e}")
                return self._fallback_single_table_analysis(df, table_name, query)
    
    def _analyze_merged_data(
        self,
        merged_df: pd.DataFrame,
        query: str,
        original_tables: List[str],
        data: Dict[str, pd.DataFrame]
    ) -> str:
        """Analyze merged DataFrame using controlled code execution."""
        
        with PandasOptionsManager():
            try:
                # Generate code prompt
                code_prompt = self._get_code_generation_prompt(
                    merged_df, 
                    query, 
                    original_tables, 
                    is_merged=True
                )
                
                # LLM generates code
                code_response = self.llm.invoke(code_prompt)
                generated_code = self._extract_output(code_response)
                
                logger.info("Generated Code:\n" + generated_code[:200] + "...")
                
                # Execute code safely
                output = self._execute_analysis_code(merged_df, generated_code)
                
                cleaned = self.response_cleaner.clean(output, original_tables)
                
                return f"{cleaned}\n\n*Integrated analysis from {len(original_tables)} tables*"
                
            except Exception as e:
                logger.error(f"Merged analysis error: {e}")
                # Fall back to separate analysis
                logger.info("Falling back to separate table analysis")
                return self._analyze_separately_and_combine(data, query)
    
    def _analyze_multiple_tables(
        self,
        data: Dict[str, pd.DataFrame],
        query: str,
        context: AnalysisContext
    ) -> str:
        """Analyze multiple tables with relationship awareness"""
        table_names = list(data.keys())
        
        # Try to merge tables if they have common columns
        merged_df = self._attempt_smart_merge(data)
        
        if merged_df is not None and not merged_df.empty:
            return self._analyze_merged_data(merged_df, query, table_names, data)
        else:
            # Fallback: analyze separately and combine
            return self._analyze_separately_and_combine(data, query)
    
    def _get_code_generation_prompt(
        self, 
        df: pd.DataFrame, 
        query: str, 
        original_tables: List[str], 
        is_merged: bool
    ) -> str:
        """Generates the prompt for the LLM to output Python code that captures output."""
        df_info = df.head(AnalysisConfig.LLM_DF_HEAD_ROWS).to_markdown()
        
        source_ref = f"from tables: {', '.join(original_tables)}" if is_merged else f"from table: {original_tables[0]}"
        
        # Get actual column names to help LLM
        columns_list = list(df.columns)
        
        prompt = f"""You are an expert data analyst. Your task is to generate executable Python code 
that uses the Pandas DataFrame named 'df' to answer the user's query. The analysis is {source_ref}.

The DataFrame 'df' has {df.shape[0]:,} rows and {df.shape[1]} columns.

AVAILABLE COLUMNS: {columns_list}

DataFrame Sample (first few rows):
{df_info}

User Query: {query}

CRITICAL RULES FOR CODE OUTPUT:
1. The DataFrame is already loaded and named 'df'. It is a pandas DataFrame.
2. Output ONLY the Python code block. DO NOT include markdown fences, explanations, or comments.
3. Use ONLY the columns listed above. Double-check column names match exactly.
4. The code MUST print the final result using print().
5. For DataFrame results: result_df.to_markdown(index=False) or result_df.to_string()
6. For single values: print(f"Result: {{value}}")
7. Always handle the case where filtering returns empty results.
8. Test that filtered data exists before accessing columns.

Example correct pattern:
filtered = df[(df['COLUMN1'] == value) & (df['COLUMN2'] == value2)]
if not filtered.empty:
    print(filtered[['COL_A', 'COL_B']].to_markdown(index=False))
else:
    print("No records found matching the criteria")

Now generate the code (ONLY code, no explanations):
"""
        return prompt

    def _execute_analysis_code(self, df: pd.DataFrame, generated_code: str) -> str:
        """Safely executes the LLM-generated Python code and captures output."""
        
        # Clean the generated code thoroughly
        generated_code = generated_code.strip()
        
        # Remove markdown code fences
        if "```python" in generated_code:
            parts = generated_code.split("```python")
            if len(parts) > 1:
                generated_code = parts[1].split("```")[0]
        elif "```" in generated_code:
            parts = generated_code.split("```")
            if len(parts) > 1:
                generated_code = parts[1] if len(parts) == 3 else parts[1].split("```")[0]
        
        generated_code = generated_code.strip()
        
        # Log the cleaned code for debugging
        logger.info(f"Cleaned code for execution:\n{generated_code}")
        
        # Prepare safe execution environment with necessary pandas functions
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "round": round,
                "abs": abs,
                "sorted": sorted,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
            },
            "pd": pd,
        }
        
        local_scope = {
            "df": df.copy(),  # Use a copy to prevent modifications
        }
        
        output_buffer = io.StringIO()
        start_time = time.time()
        
        try:
            # Execute with stdout capture
            with contextlib.redirect_stdout(output_buffer):
                exec(generated_code, safe_globals, local_scope)
            
            # Get the printed output
            result = output_buffer.getvalue().strip()
            
            execution_time = time.time() - start_time
            logger.info(f"Code executed successfully in {execution_time:.2f}s")
            
            # Handle empty results
            if not result:
                logger.warning("Code executed but produced no output")
                # Check if there's a result variable in local scope
                if 'result' in local_scope:
                    result = str(local_scope['result'])
                elif 'filtered' in local_scope and hasattr(local_scope['filtered'], 'to_markdown'):
                    try:
                        result = local_scope['filtered'].to_markdown(index=False)
                    except:
                        result = str(local_scope['filtered'])
                else:
                    return "The analysis completed but returned no results. This might mean no data matched the criteria."
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_traceback = traceback.format_exc()
            logger.error(f"Code execution failed in {execution_time:.2f}s.\nCode:\n{generated_code}\n\nError: {e}\n\nTraceback:\n{error_traceback}")
            
            # Provide helpful error messages
            error_msg = str(e)
            if "'NoneType' object" in error_msg:
                logger.error("NoneType error - likely trying to access attributes on None")
                return "The analysis encountered an issue: unable to process the data with the generated query. The filtered dataset might be empty or a column reference is incorrect."
            elif "KeyError" in error_msg:
                # Extract the missing key if possible
                return f"Column not found in dataset. Please verify the column names. Error: {error_msg}"
            elif "AttributeError" in error_msg:
                return f"Attribute error in analysis: {error_msg}"
            else:
                # For debugging, include the actual error
                return f"Analysis execution failed: {error_msg}\n\nPlease try rephrasing your query or check if the referenced columns exist."
    
    def _get_single_table_analysis_prompt(self, table_name: str, df: pd.DataFrame, query: str) -> str:
        """Generates a combined prompt for the single table case (direct answer)."""
        df_info = df.head(AnalysisConfig.LLM_DF_HEAD_ROWS).to_markdown()
        
        return f"""You are an expert data analyst. Your task is to provide a comprehensive, direct answer 
to the user's query based ONLY on the provided data.

Analysis Target: '{table_name}' table with {df.shape[0]:,} rows and {df.shape[1]} columns.

DataFrame Sample/Schema (Pandas Head):
{df_info}

Columns: {list(df.columns)}

User Query: {query}

Provide a clear, comprehensive answer. Include specific examples and statistics when relevant.
Do NOT use or generate any Python code. Your entire response must be the final answer.
"""
    
    def _attempt_smart_merge(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Attempt to intelligently merge multiple DataFrames."""
        try:
            table_names = list(data.keys())
            if len(table_names) < 2:
                return None
            
            # Start with first table
            result = data[table_names[0]].copy()
            logger.info(f"Starting merge with {table_names[0]}: {result.shape}, columns: {list(result.columns)}")
            
            for i in range(1, len(table_names)):
                next_table = data[table_names[i]].copy()
                logger.info(f"Next table {table_names[i]}: {next_table.shape}, columns: {list(next_table.columns)}")
                
                # Find common columns
                common_cols = list(set(result.columns) & set(next_table.columns))
                
                if not common_cols:
                    logger.warning(f"No common columns for merge with {table_names[i]}")
                    return None
                
                logger.info(f"Merging on columns: {common_cols}")
                
                # Handle data type mismatches for merge columns
                for col in common_cols:
                    try:
                        # Ensure both columns have compatible types
                        if result[col].dtype != next_table[col].dtype:
                            logger.info(f"Type mismatch for {col}: {result[col].dtype} vs {next_table[col].dtype}, converting to string")
                            result[col] = result[col].astype(str)
                            next_table[col] = next_table[col].astype(str)
                    except Exception as e:
                        logger.warning(f"Type conversion warning for {col}: {e}")
                        result[col] = result[col].astype(str)
                        next_table[col] = next_table[col].astype(str)
                
                # Perform left join to preserve all records from primary table
                result = pd.merge(
                    result, 
                    next_table, 
                    on=common_cols, 
                    how='left', 
                    suffixes=('', f'_{table_names[i]}')
                )
                
                logger.info(f"After merge with {table_names[i]}: {result.shape}, columns: {list(result.columns)}")
            
            if result.empty:
                logger.warning("Merge resulted in empty DataFrame")
                return None
            
            # Log final merged DataFrame info
            logger.info(f"Final merged DataFrame: {result.shape} rows, columns: {list(result.columns)}")
            logger.info(f"Sample of merged data:\n{result.head(2)}")
            
            return result.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            return None
    
    def _analyze_separately_and_combine(
        self,
        data: Dict[str, pd.DataFrame],
        query: str
    ) -> str:
        """Fallback: analyze tables separately and combine results."""
        results = []
        
        for table_name, df in data.items():
            try:
                result = self._analyze_single_table({table_name: df}, query)
                results.append(f"**From {table_name}:**\n{result}")
            except Exception as e:
                logger.error(f"Error analyzing {table_name}: {e}")
                results.append(f"**From {table_name}:** Unable to analyze")
        
        return "\n\n".join(results)
    
    def _extract_output(self, response) -> str:
        """Extract text output from LLM response."""
        if isinstance(response, dict) and 'output' in response:
            return response['output']
        elif hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _fallback_single_table_analysis(
        self,
        df: pd.DataFrame,
        table_name: str,
        query: str
    ) -> str:
        """Provide basic statistics when analysis fails."""
        try:
            summary = f"**Analysis of {table_name}**\n\n"
            summary += f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
            
            summary += "**Columns:**\n"
            for col in df.columns[:10]:
                dtype = df[col].dtype
                non_null = df[col].notna().sum()
                summary += f"- {col} ({dtype}): {non_null:,} values\n"
            
            numeric_cols = df.select_dtypes(include=['number']).columns[:5]
            if len(numeric_cols) > 0:
                summary += f"\n**Sample Statistics:**\n"
                for col in numeric_cols:
                    summary += f"- {col}: mean={df[col].mean():.2f}, "
                    summary += f"min={df[col].min():.2f}, max={df[col].max():.2f}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return f"Unable to analyze {table_name} due to processing error."