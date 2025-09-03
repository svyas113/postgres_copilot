import os
import sys
from typing import Optional, Dict, Any, List, Tuple

# Import with proper fallback pattern
try:
    from . import memory_module
    from . import vector_store_module
    from .llm_client import LLMClient
    from .error_handler_module import handle_exception
except ImportError:
    try:
        import memory_module
        import vector_store_module
        from llm_client import LLMClient
        from error_handler_module import handle_exception
    except ImportError:
        # Try importing from parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from application import memory_module
        from application import vector_store_module
        from application.llm_client import LLMClient
        from application.error_handler_module import handle_exception

class SchemaNavigationModule:
    def __init__(self, db_name_identifier: str):
        self.db_name_identifier = db_name_identifier
        
    def _get_schema_context(self) -> str:
        """
        Retrieves the schema backup content for the current database.
        """
        try:
            # Get the schema backup filepath using memory_module
            backup_filepath = memory_module.get_schema_backup_filepath(self.db_name_identifier)
            
            # Check if the file exists
            if not os.path.exists(backup_filepath):
                return f"Schema backup file not found for database '{self.db_name_identifier}'."
            
            # Read the schema backup file
            with open(backup_filepath, 'r', encoding='utf-8') as f:
                schema_content = f.read()
            
            return schema_content
        except Exception as e:
            handle_exception(e, user_query=f"get_schema_context for {self.db_name_identifier}")
            return f"Error retrieving schema context: {str(e)}"
    
    async def answer_schema_question(self, question: str, llm_client: LLMClient, previous_answer: Optional[str] = None) -> str:
        """
        Answers a natural language question about the database schema.
        
        Args:
            question: The natural language question about the schema
            llm_client: The LLM client instance
            previous_answer: Optional previous answer for context in follow-up questions
            
        Returns:
            The answer to the question
        """
        try:
            # Get the schema context
            schema_context = self._get_schema_context()
            
            # Prepare the prompt for the LLM
            prompt = self._prepare_prompt(question, schema_context, previous_answer)
            
            # Send the prompt to the LLM
            messages = [{"role": "user", "content": prompt}]
            response = await llm_client.chat(messages)
            
            # Extract the answer from the response
            if response and response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                return answer
            else:
                return "I couldn't generate an answer to your question about the database schema."
        except Exception as e:
            handle_exception(e, user_query=question, context={"db_name": self.db_name_identifier})
            return f"Error answering schema question: {str(e)}"
    
    def _prepare_prompt(self, question: str, schema_context: str, previous_answer: Optional[str] = None) -> str:
        """
        Prepares the prompt for the LLM.
        
        Args:
            question: The natural language question about the schema
            schema_context: The schema context from the backup file
            previous_answer: Optional previous answer for context in follow-up questions
            
        Returns:
            The formatted prompt
        """
        # Base prompt
        prompt = f"""You are a helpful database assistant that answers questions about database schemas.

DATABASE SCHEMA INFORMATION:
```
{schema_context[:50000]}  # Limit context size to avoid token limits
```

USER QUESTION: {question}
"""

        # Add previous answer context if available
        if previous_answer:
            prompt += f"""

PREVIOUS ANSWER: 
{previous_answer}

Please answer the follow-up question based on the schema information and the previous answer.
"""
        else:
            prompt += """

Please answer the question based on the schema information. Be concise but thorough, focusing on the relevant tables, columns, and relationships.
"""

        return prompt
