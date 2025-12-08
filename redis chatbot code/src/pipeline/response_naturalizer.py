import logging
import re
from typing import Optional
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class ResponseNaturalizer:
    """
    Post-processes technical analysis responses into natural language.
    Makes responses more conversational and user-friendly.
    """
    
    def __init__(self, llm: VertexAI):
        """
        Initialize with LLM instance.
        
        Args:
            llm: VertexAI LLM instance for naturalizing responses
        """
        self.llm = llm
    
    def naturalize_response(
        self, 
        raw_response: str, 
        user_query: str,
        max_retries: int = 2
    ) -> str:
        """
        Convert technical response into natural, conversational language.
        
        Args:
            raw_response: Raw technical response from analyzer
            user_query: Original user query for context
            max_retries: Maximum retry attempts if naturalization fails
            
        Returns:
            Naturalized response in conversational language
        """
        try:
            # Pre-clean the response
            cleaned_response = self._pre_clean_response(raw_response)
            
            # If response is already short and natural, return as-is
            if len(cleaned_response) < 500 and not self._has_technical_formatting(cleaned_response):
                return cleaned_response
            
            # Use LLM to naturalize
            for attempt in range(max_retries):
                try:
                    naturalized = self._llm_naturalize(cleaned_response, user_query)
                    
                    # Validate the output
                    if self._is_valid_naturalized_response(naturalized):
                        logger.info("Successfully naturalized response")
                        return naturalized
                    else:
                        logger.warning(f"Naturalization attempt {attempt + 1} produced invalid output")
                        
                except Exception as e:
                    logger.error(f"Naturalization attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        # Last attempt failed, return cleaned version
                        return cleaned_response
            
            # All attempts failed, return cleaned version
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error in naturalize_response: {e}", exc_info=True)
            # Return original if all else fails
            return raw_response
    
    def _pre_clean_response(self, response: str) -> str:
        """
        Pre-clean response by removing obvious technical artifacts.
        
        Args:
            response: Raw response text
            
        Returns:
            Pre-cleaned response
        """
        cleaned = response
        
        # Remove analysis metadata lines
        patterns_to_remove = [
            r'\*Analysis of .+ \(\d{1,3}(,\d{3})* rows\)\*',
            r'\*Integrated analysis from \d+ tables?\*',
            r'\*Combined analysis from \d+ tables?\*',
            r'---+',  # Remove separator lines
            r'^\*\*Dataset:\*\* \d{1,3}(,\d{3})* rows × \d+ columns$',
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _has_technical_formatting(self, text: str) -> bool:
        """
        Check if text contains technical formatting that needs naturalization.
        
        Args:
            text: Text to check
            
        Returns:
            True if text has technical formatting
        """
        technical_indicators = [
            r'\*\*Columns:\*\*',
            r'\*\*Sample Statistics:\*\*',
            r'Dataset:.*rows × \d+ columns',
            r'dtype:',
            r'mean=.*min=.*max=',
            r'EMP_CODE \(object\):',
            r'- \w+ \(\w+\): \d{1,3}(,\d{3})* values'
        ]
        
        for indicator in technical_indicators:
            if re.search(indicator, text):
                return True
        
        return False
    
    def _get_naturalization_prompt(self) -> str:
        """
        Get the system prompt for naturalizing responses.
        
        Returns:
            System prompt string
        """
        return """You are a helpful assistant that converts technical database analysis outputs into natural, conversational responses.

Your task:
1. Convert technical formatting (bullet points, statistics, column names) into natural sentences
2. Focus on answering the user's specific question
3. Present information in a friendly, conversational tone
4. Keep the most relevant information from the technical output
5. Remove redundant or overly technical details (like dtype, exact row counts, etc.)
6. If multiple tables were analyzed, synthesize the information into a cohesive narrative

Guidelines:
- Write in complete, natural sentences
- Use "they/their" or appropriate pronouns instead of "employee with EMP_CODE..."
- Convert dates to readable format (e.g., "March 3, 2025" instead of "2025-03-03")
- Focus on answering what the user actually asked
- Be concise but informative
- Do NOT include metadata like "Analysis of X table" or row counts
- Do NOT use markdown formatting (**, *, ---)
- Write as if you're having a conversation with the user

Example transformations:
Technical: "* **Employee Name:** SUDHARSHAN MA * **Basic Pay:** 15600"
Natural: "The employee's name is Sudharshan MA and they have a basic pay of 15,600."

Technical: "Dataset: 86,098 rows × 30 columns"
Natural: [Remove this information entirely unless relevant to the query]

Your response should read naturally, as if a knowledgeable person is answering the user's question directly."""
    
    def _llm_naturalize(self, cleaned_response: str, user_query: str) -> str:
        """
        Use LLM to naturalize the response.
        
        Args:
            cleaned_response: Pre-cleaned technical response
            user_query: Original user query
            
        Returns:
            Naturalized response
        """
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_naturalization_prompt()),
                ("human", """User's Question: {user_query}

Technical Response to Naturalize:
{technical_response}

Please provide a natural, conversational response that directly answers the user's question.""")
            ])
            
            chain = prompt_template | self.llm
            
            response = chain.invoke({
                "user_query": user_query,
                "technical_response": cleaned_response
            })
            
            # Extract text from response
            if hasattr(response, 'content'):
                naturalized = response.content.strip()
            else:
                naturalized = str(response).strip()
            
            return naturalized
            
        except Exception as e:
            logger.error(f"LLM naturalization failed: {e}")
            raise
    
    def _is_valid_naturalized_response(self, response: str) -> bool:
        """
        Validate that the naturalized response is acceptable.
        
        Args:
            response: Naturalized response to validate
            
        Returns:
            True if response is valid
        """
        if not response or len(response.strip()) < 20:
            return False
        
        # Check that it doesn't still have heavy technical formatting
        if response.count('**') > 5:  # Some markdown is ok, but not excessive
            return False
        
        if response.count('*Analysis of') > 0:
            return False
        
        if 'Dataset:' in response and 'rows ×' in response:
            return False
        
        return True


class EnhancedSafeDataFrameAnalyzer:
    """
    Enhanced analyzer with built-in response naturalization.
    Extends SafeDataFrameAnalyzer with post-processing.
    """
    
    def __init__(self, llm: VertexAI, enable_naturalization: bool = True):
        """
        Initialize enhanced analyzer.
        
        Args:
            llm: VertexAI LLM instance
            enable_naturalization: Whether to enable response naturalization
        """
        self.llm = llm
        self.enable_naturalization = enable_naturalization
        
        # Initialize the base analyzer components
        from src.pipeline.core import ResponseCleaner
        self.response_cleaner = ResponseCleaner()
        
        # Initialize naturalizer if enabled
        if self.enable_naturalization:
            self.naturalizer = ResponseNaturalizer(llm)
            logger.info("Response naturalization enabled")
        else:
            self.naturalizer = None
            logger.info("Response naturalization disabled")
    
    def analyze_safe(
        self,
        data,
        query: str,
        context,
        naturalize: bool = None
    ) -> str:
        """
        Analyze data and optionally naturalize the response.
        
        Args:
            data: Dictionary of DataFrames to analyze
            query: User query
            context: Analysis context
            naturalize: Override naturalization setting for this call
            
        Returns:
            Analysis response (naturalized if enabled)
        """
        # Import here to avoid circular imports
        from src.pipeline.safe_analyzer import SafeDataFrameAnalyzer
        
        # Use base analyzer to get raw response
        base_analyzer = SafeDataFrameAnalyzer(self.llm)
        raw_response = base_analyzer.analyze_safe(data, query, context)
        
        # Determine if we should naturalize
        should_naturalize = (
            naturalize if naturalize is not None 
            else self.enable_naturalization
        )
        
        if should_naturalize and self.naturalizer:
            try:
                naturalized_response = self.naturalizer.naturalize_response(
                    raw_response,
                    query
                )
                return naturalized_response
            except Exception as e:
                logger.error(f"Naturalization failed, returning raw response: {e}")
                return raw_response
        
        return raw_response