import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import ValidationError

# Import the hyde_feedback_module
try:
    from . import hyde_feedback_module
except ImportError:
    try:
        import hyde_feedback_module
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application import hyde_feedback_module
        except ImportError as e:
            print(f"Error importing hyde_feedback_module: {e}", file=sys.stderr)
            hyde_feedback_module = None

# Import necessary modules
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import application modules
try:
    from .pydantic_models import SQLGenerationResponse, FeedbackIteration
except ImportError:
    try:
        from pydantic_models import SQLGenerationResponse, FeedbackIteration
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application.pydantic_models import SQLGenerationResponse, FeedbackIteration
        except ImportError as e:
            print(f"Error importing pydantic_models: {e}", file=sys.stderr)
            # Define minimal fallback classes if needed
            from pydantic import BaseModel
            class SQLGenerationResponse(BaseModel):
                sql_query: Optional[str] = None
                explanation: Optional[str] = None
                error_message: Optional[str] = None
            
            class FeedbackIteration(BaseModel):
                user_feedback_text: str
                corrected_sql_attempt: str
                corrected_explanation: str

class SQLFeedbackProcessor:
    """
    Handles the processing of user feedback on SQL queries and generates revised queries.
    """
    
    def __init__(self, db_name_identifier: str):
        """
        Initialize the SQL feedback processor with the database identifier.
        
        Args:
            db_name_identifier: The identifier for the database.
        """
        self.db_name_identifier = db_name_identifier
        self.MAX_FEEDBACK_RETRIES = 3  # Number of retries for the LLM to fix its output
        self.MAX_SQL_EXECUTION_RETRIES = 5  # Number of retries for the LLM to fix SQL execution errors
    
    @staticmethod
    def _extract_json_from_response(text: str) -> Optional[str]:
        """
        Extracts a JSON object from a string, even if it's embedded in other text.
        Handles markdown code blocks.
        """
        # Regex to find a JSON object within ```json ... ``` or just { ... }
        match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
        if match:
            # Prioritize the content of ```json ... ``` if present
            return match.group(1) if match.group(1) else match.group(2)
        return None
        
    @staticmethod
    def _extract_tables_from_query(sql: str) -> List[str]:
        """
        Extracts table names from a SQL query using a simple regex.
        Looks for tables after FROM and JOIN clauses.
        """
        # This regex finds words that follow 'FROM' or 'JOIN' keywords.
        # It's a simple approach and might need refinement for complex cases (e.g., subqueries, schemas).
        pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)\b'
        tables = re.findall(pattern, sql, re.IGNORECASE)
        return list(set(tables))  # Return unique table names
    
    async def process_feedback(
        self,
        natural_language_question: str,
        current_sql: str,
        current_explanation: str,
        feedback_text: str,
        feedback_iterations: List[Dict[str, str]],
        schema_and_sample_data: Dict[str, Any],
        insights_markdown_content: Optional[str],
        llm_client: Any
    ) -> Dict[str, Any]:
        """
        Process user feedback on a SQL query and generate a revised query.
        
        Args:
            natural_language_question: The original natural language question.
            current_sql: The current SQL query.
            current_explanation: The explanation for the current SQL query.
            feedback_text: The user's feedback on the SQL query.
            feedback_iterations: List of previous feedback iterations.
            schema_and_sample_data: The database schema and sample data.
            insights_markdown_content: The current insights markdown content.
            llm_client: The LLM client for generating revised SQL.
            
        Returns:
            A dictionary with the result of the feedback processing.
        """
        try:
            # Prepare schema context using HyDE feedback module
            schema_context_str = "No schema or sample data provided."
            table_names_from_hyde = []
            
            if schema_and_sample_data and hyde_feedback_module:
                try:
                    # Use the hyde_feedback_module to get focused schema context
                    hyde_context, table_names_from_hyde = await hyde_feedback_module.retrieve_hyde_feedback_context(
                        nlq=natural_language_question,
                        current_sql=current_sql,
                        feedback_text=feedback_text,
                        db_name_identifier=self.db_name_identifier,
                        llm_client=llm_client
                    )
                    
                    # If HyDE context is available and not an error message, use it
                    if hyde_context and "Failed" not in hyde_context and "Error" not in hyde_context:
                        schema_context_str = hyde_context
                        print(f"Using HyDE feedback context with {len(table_names_from_hyde)} tables")
                    else:
                        # Fallback to simplified approach if HyDE fails
                        table_names = list(schema_and_sample_data.keys())[:5]  # Limit to first 5 tables
                        schema_context_str = f"Available tables include: {', '.join(table_names)}"
                        print(f"Using fallback schema context: {schema_context_str}")
                except Exception as e:
                    # Fallback to simplified approach if HyDE fails
                    table_names = list(schema_and_sample_data.keys())[:5]  # Limit to first 5 tables
                    schema_context_str = f"Available tables include: {', '.join(table_names)}"
                    print(f"Error using HyDE feedback module: {e}. Using fallback schema context.")
            elif schema_and_sample_data:
                # Fallback if hyde_feedback_module is not available
                try:
                    table_names = list(schema_and_sample_data.keys())[:5]  # Limit to first 5 tables
                    schema_context_str = f"Available tables include: {', '.join(table_names)}"
                except TypeError as e:
                    schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
            
            # Prepare insights context
            insights_context_str = "No cumulative insights provided."
            if insights_markdown_content and insights_markdown_content.strip():
                insights_context_str = insights_markdown_content
            
            # Prepare feedback history context
            feedback_history_str = "No previous feedback iterations."
            if feedback_iterations:
                feedback_parts = ["Previous feedback iterations:"]
                for i, iteration in enumerate(feedback_iterations):
                    feedback_parts.append(f"Iteration {i+1}:")
                    feedback_parts.append(f"  User Feedback: \"{iteration['user_feedback_text']}\"")
                    feedback_parts.append(f"  Revised SQL: ```sql\n{iteration['corrected_sql_attempt']}\n```")
                    feedback_parts.append(f"  Explanation: {iteration['corrected_explanation']}")
                feedback_history_str = "\n".join(feedback_parts)
            
            # Prepare the prompt for the LLM
            prompt = (
                f"You are an expert PostgreSQL SQL writer. A user has provided feedback on a SQL query that was generated "
                f"for their natural language question. Your task is to revise the SQL query based on the feedback.\n\n"
                f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                f"CURRENT SQL QUERY:\n```sql\n{current_sql}\n```\n\n"
                f"CURRENT EXPLANATION: {current_explanation}\n\n"
                f"USER FEEDBACK: \"{feedback_text}\"\n\n"
                f"{feedback_history_str}\n\n"
                f"RELEVANT DATABASE SCHEMA INFORMATION:\n```\n{schema_context_str}\n```\n\n"
                f"CUMULATIVE INSIGHTS:\n```markdown\n{insights_context_str}\n```\n\n"
                f"Based on the user's feedback, revise the SQL query. Your response MUST be a single, valid JSON object "
                f"with the following structure: {{ \"sql_query\": \"<Your revised SQL query>\", \"explanation\": \"<Your explanation>\" }}\n\n"
                f"The explanation should describe how the revised query addresses the user's feedback and how it answers "
                f"the original natural language question. The SQL query must start with SELECT."
            )
            
            # Variables to store the LLM response
            llm_response_text = ""
            parsed_response = None
            last_error_feedback_to_llm = ""
            
            # Initial messages list for LiteLLM
            messages_for_llm = []
            if hasattr(llm_client, 'conversation_history'):
                messages_for_llm = llm_client.conversation_history[:]
            
            # Retry loop for LLM calls
            for attempt in range(self.MAX_FEEDBACK_RETRIES + 1):
                user_message_content = ""
                if attempt > 0 and last_error_feedback_to_llm:  # This implies a retry
                    user_message_content = (
                        f"{last_error_feedback_to_llm}\n\n"
                        f"Based on the previous error, please try again.\n"
                        f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                        f"CURRENT SQL QUERY:\n```sql\n{current_sql}\n```\n\n"
                        f"USER FEEDBACK: \"{feedback_text}\"\n\n"
                        f"RELEVANT DATABASE SCHEMA:\n```\n{schema_context_str}\n```\n\n"
                        f"Respond ONLY with a single JSON object matching the structure: "
                        f"{{ \"sql_query\": \"<Your revised SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                        f"Ensure the SQL query strictly starts with 'SELECT'."
                    )
                else:  # First attempt
                    user_message_content = prompt
                
                # Add the current user prompt to messages for this specific call
                current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]
                
                try:
                    # Send the message to the LLM
                    if hasattr(llm_client, '_send_message_to_llm'):
                        # Use the client's method if available
                        llm_response_obj = await llm_client._send_message_to_llm(
                            current_call_messages, 
                            natural_language_question, 
                            0, 
                            source='sql_feedback'
                        )
                        llm_response_text, tool_calls_made = await llm_client._process_llm_response(llm_response_obj)
                    else:
                        # Fallback to a simpler approach
                        llm_response = await llm_client.chat(current_call_messages)
                        llm_response_text = llm_response.choices[0].message.content
                        tool_calls_made = False
                    
                    if tool_calls_made:
                        last_error_feedback_to_llm = "Your response included an unexpected tool call. Please provide the JSON response directly."
                        if attempt < self.MAX_FEEDBACK_RETRIES:
                            continue
                        else:
                            raise Exception("LLM attempted tool call instead of providing JSON for SQL revision.")
                    
                    json_from_response = self._extract_json_from_response(llm_response_text)
                    
                    if not json_from_response:
                        last_error_feedback_to_llm = "Your response did not contain a valid JSON object. Please provide the JSON output as instructed."
                        if attempt < self.MAX_FEEDBACK_RETRIES:
                            continue
                        else:
                            raise Exception("LLM failed to provide a JSON response for SQL revision.")
                    
                    try:
                        parsed_response = SQLGenerationResponse.model_validate_json(json_from_response)
                    except ValidationError as ve:
                        last_error_feedback_to_llm = f"Your response contained invalid JSON: {str(ve)}. Please provide a valid JSON object."
                        if attempt < self.MAX_FEEDBACK_RETRIES:
                            continue
                        else:
                            raise Exception(f"LLM failed to provide valid JSON after retries: {str(ve)}")
                    
                    if not parsed_response.sql_query:
                        error_detail = parsed_response.error_message or "LLM did not provide an SQL query in the 'sql_query' field."
                        last_error_feedback_to_llm = (
                            f"Your previous attempt did not produce an SQL query in the 'sql_query' field. LLM message: '{error_detail}'. "
                            f"Ensure 'sql_query' field is populated with a valid SQL string."
                        )
                        if attempt < self.MAX_FEEDBACK_RETRIES:
                            continue
                        else:
                            raise Exception(f"LLM failed to provide SQL query after retries. Last message: {error_detail}")
                    
                    if not parsed_response.sql_query.strip().upper().startswith("SELECT"):
                        last_error_feedback_to_llm = (
                            f"Your generated SQL query was: ```sql\n{parsed_response.sql_query}\n```\n"
                            f"This query does not start with SELECT, which is a requirement. Please regenerate a valid SELECT query."
                        )
                        if attempt < self.MAX_FEEDBACK_RETRIES:
                            parsed_response = None  # Invalidate this response
                            continue
                        else:
                            raise Exception("LLM failed to generate a SELECT query after retries.")
                    
                    break
                
                except Exception as e:
                    last_error_feedback_to_llm = f"An error occurred: {str(e)}. Please try again."
                    if attempt == self.MAX_FEEDBACK_RETRIES:
                        return {
                            "success": False,
                            "message": f"Failed to process feedback after all attempts: {str(e)}"
                        }
            
            if not parsed_response or not parsed_response.sql_query:
                return {
                    "success": False,
                    "message": "Failed to generate a revised SQL query based on your feedback."
                }
            
            # Execute SQL with retry logic for execution errors
            revised_sql = parsed_response.sql_query
            revised_explanation = parsed_response.explanation or "No explanation provided."
            original_sql = revised_sql
            original_explanation = revised_explanation
            
            execution_result = None
            execution_error = None
            row_limit_for_preview = 1  # Show only 1 row of execution result
            
            # Execute SQL with retry logic for execution errors
            for exec_attempt in range(self.MAX_SQL_EXECUTION_RETRIES + 1):
                try:
                    # Execute the SQL query using the client's session if available
                    if hasattr(llm_client, 'session') and llm_client.session:
                        # Use the session directly from the LLM client
                        try:
                            exec_result_obj = await llm_client.session.call_tool(
                                "execute_postgres_query",
                                {"query": revised_sql, "row_limit": row_limit_for_preview}
                            )
                        except Exception as e:
                            raise
                        
                        if hasattr(llm_client, '_extract_mcp_tool_call_output'):
                            raw_exec_output = llm_client._extract_mcp_tool_call_output(exec_result_obj, "execute_postgres_query")
                        else:
                            raw_exec_output = exec_result_obj
                        
                        try:
                            # Attempt to parse the output if it's a JSON string
                            if isinstance(raw_exec_output, str):
                                exec_data = json.loads(raw_exec_output)
                            else:
                                exec_data = raw_exec_output

                            # Check for error in the structured response
                            if isinstance(exec_data, dict) and exec_data.get("status") == "error":
                                execution_error = exec_data.get("message", "Unknown execution error.")
                                
                                if exec_attempt < self.MAX_SQL_EXECUTION_RETRIES:
                                    # --- Intelligent Retry: Fetch Correct Schema ---
                                    detailed_schema_context = ""
                                    try:
                                        tables_in_query = self._extract_tables_from_query(revised_sql)
                                        if tables_in_query:
                                            table_schemas = []
                                            for table_name in tables_in_query:
                                                try:
                                                    table_info_obj = await llm_client.session.call_tool(
                                                        "describe_table", {"table_name": table_name}
                                                    )
                                                    if hasattr(llm_client, '_extract_mcp_tool_call_output'):
                                                        table_info = llm_client._extract_mcp_tool_call_output(table_info_obj, "describe_table")
                                                    else:
                                                        table_info = table_info_obj
                                                    table_schemas.append(json.dumps({table_name: table_info}, indent=2))
                                                except Exception as e_desc:
                                                    print(f"Error describing table {table_name}: {e_desc}", file=sys.stderr)
                                            detailed_schema_context = "\n".join(table_schemas)
                                    except Exception as e_extract:
                                        print(f"Error extracting tables from query: {e_extract}", file=sys.stderr)
                                    # --- End Intelligent Retry ---

                                    fix_user_message_content = (
                                        f"The previously generated SQL query resulted in an execution error.\n"
                                        f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                                        f"USER FEEDBACK: \"{feedback_text}\"\n\n"
                                        f"SQL QUERY WITH ERROR:\n```sql\n{revised_sql}\n```\n\n"
                                        f"EXECUTION ERROR MESSAGE:\n{execution_error}\n\n"
                                        f"To help you fix this, here is the precise, correct schema for the table(s) involved in the query:\n"
                                        f"CORRECT AND DETAILED SCHEMA:\n```json\n{detailed_schema_context}\n```\n\n"
                                        f"Please provide a corrected SQL query using ONLY the columns listed in the schema above. For the explanation, describe how the *corrected* SQL query answers the original question. Do not mention the error or the process of fixing it.\n"
                                        f"Respond ONLY with a single, valid JSON object matching this structure: "
                                        f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}"
                                    )
                                    
                                    fix_call_messages = messages_for_llm + [{"role": "user", "content": fix_user_message_content}]
                                    
                                    try:
                                        # Send the message to the LLM
                                        if hasattr(llm_client, '_send_message_to_llm'):
                                            fix_llm_response_obj = await llm_client._send_message_to_llm(fix_call_messages, natural_language_question, 0, source='sql_feedback')
                                            fix_text, fix_tool_calls_made = await llm_client._process_llm_response(fix_llm_response_obj)
                                        else:
                                            fix_llm_response = await llm_client.chat(fix_call_messages)
                                            fix_text = fix_llm_response.choices[0].message.content
                                            fix_tool_calls_made = False

                                        if fix_tool_calls_made:
                                            continue

                                        json_from_fix = self._extract_json_from_response(fix_text)
                                        
                                        if not json_from_fix:
                                            continue

                                        try:
                                            fixed_response = SQLGenerationResponse.model_validate_json(json_from_fix)
                                            
                                            if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                                revised_sql = fixed_response.sql_query
                                                revised_explanation = fixed_response.explanation
                                                execution_error = None 
                                                continue
                                        except ValidationError:
                                            continue
                                    except Exception as fix_e:
                                        print(f"Error during LLM SQL fix attempt: {fix_e}", file=sys.stderr)
                            else:
                                execution_result = exec_data
                                execution_error = None
                                break
                        except json.JSONDecodeError:
                            # If output is not a valid JSON, treat it as a potential success or non-error message
                            execution_result = raw_exec_output
                            execution_error = None
                            break
                    else:
                        # If no session is available, we can't execute the query
                        execution_error = "No database session available to execute the query."
                        break
                        
                except Exception as e:
                    execution_error = str(e)
                    print(f"Error during SQL execution (attempt {exec_attempt + 1}): {e}", file=sys.stderr)
                    if exec_attempt < self.MAX_SQL_EXECUTION_RETRIES:
                        continue
            
            if execution_error and revised_sql != original_sql:
                # Revert to original SQL for display if all fix attempts failed
                revised_sql = original_sql
                revised_explanation = original_explanation
            
            # Construct the success message
            message = f"Revised SQL based on your feedback:\n```sql\n{revised_sql}\n```\n\nExplanation: {revised_explanation}\n\n"
            
            if execution_error:
                message += f"Execution Error: {execution_error}\n\n"
            elif execution_result is not None:
                # Format the execution result for display
                if isinstance(execution_result, dict) and execution_result.get("status") == "success":
                    data = execution_result.get("data")
                    if data and isinstance(data, list) and len(data) > 0:
                        message += f"Execution Result (first {min(len(data), row_limit_for_preview)} rows):\n"
                        for i, row in enumerate(data[:row_limit_for_preview]):
                            message += f"{row}\n"
                        if len(data) > row_limit_for_preview:
                            message += f"... and {len(data) - row_limit_for_preview} more rows\n"
                    else:
                        message += "Query executed successfully, but no rows were returned.\n"
                else:
                    message += f"Execution Result: {execution_result}\n"
            
            message += "\nIf this is correct, use '/approve'. If you want to provide more feedback, use '/feedback Your feedback text'."
            
            return {
                "success": True,
                "message": message,
                "revised_sql": revised_sql,
                "revised_explanation": revised_explanation,
                "execution_result": execution_result,
                "execution_error": execution_error
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during feedback processing: {e}"
            }
