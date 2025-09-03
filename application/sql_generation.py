import json
import re
import sys
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import os
from datetime import datetime

# Import necessary modules for SQL generation
from pydantic import ValidationError, BaseModel

# Try to import pydantic_models with proper fallback
try:
    from .pydantic_models import SQLGenerationResponse
except ImportError:
    try:
        from pydantic_models import SQLGenerationResponse
    except ImportError:
        print(f"Error importing pydantic_models", file=sys.stderr)
        # Define a simple fallback SQLGenerationResponse class
        class SQLGenerationResponse(BaseModel):
            sql_query: Optional[str] = None
            explanation: Optional[str] = None
            error_message: Optional[str] = None

# Import modules using relative imports for application modules with proper fallbacks
# For each module, try relative import first, then absolute import, then define fallbacks
try:
    from . import hyde_module
except ImportError:
    try:
        import hyde_module
    except ImportError:
        print(f"Error importing hyde_module", file=sys.stderr)
        hyde_module = None

try:
    from . import vector_store_module
except ImportError:
    try:
        import vector_store_module
    except ImportError:
        print(f"Error importing vector_store_module", file=sys.stderr)
        vector_store_module = None

try:
    from . import memory_module
except ImportError:
    try:
        import memory_module
    except ImportError:
        print(f"Error importing memory_module", file=sys.stderr)
        memory_module = None

try:
    from . import join_path_finder
except ImportError:
    try:
        import join_path_finder
    except ImportError:
        print(f"Error importing join_path_finder", file=sys.stderr)
        join_path_finder = None

try:
    from . import schema_vectorization_module
except ImportError:
    try:
        import schema_vectorization_module
    except ImportError:
        print(f"Error importing schema_vectorization_module", file=sys.stderr)
        schema_vectorization_module = None

# Ensure all required directories exist
def ensure_logs_directory():
    """Ensure the logs directory exists for Hyde module logging."""
    try:
        if memory_module:
            memory_base_path = memory_module.get_memory_base_path()
            logs_dir = os.path.join(memory_base_path, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            print(f"Logs directory ensured at: {logs_dir}")
            return True
    except Exception as e:
        print(f"Error ensuring logs directory: {e}", file=sys.stderr)
    return False

def _log_sql_generation_debug(db_name_identifier: str, event: str, data: Dict[str, Any]) -> None:
    """
    Logs SQL generation debug information to a JSON file.
    
    Args:
        db_name_identifier: The database identifier
        event: The event name (e.g., 'hyde_started', 'hyde_failed', 'context_selection')
        data: Dictionary containing debug data
    """
    try:
        # Get the logs directory path - use hardcoded path to avoid dependency on memory_module
        logs_dir = os.path.join('/home/shivam/fastworkflow/data/memory/logs')
        
        # Ensure logs directory exists
        os.makedirs(logs_dir, exist_ok=True)
        
        # Prepare log file path
        log_file_path = os.path.join(logs_dir, 'sql_generation_debug.json')
        
        # Create log entry with timestamp
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "db_name_identifier": db_name_identifier,
            "event": event,
            **data  # Include all data fields
        }
        
        # Append to log file
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
    except Exception as e:
        print(f"Error logging SQL generation debug: {e}", file=sys.stderr)

class SQLGenerator:
    """
    A class that handles SQL generation from natural language questions.
    Implements the architecture described in architecture_detailed.md.
    """
    
    def __init__(self, db_name_identifier: str):
        """
        Initialize the SQL Generator.
        
        Args:
            db_name_identifier: Identifier for the database being queried
        """
        self.db_name_identifier = db_name_identifier
        self.MAX_SQL_GEN_RETRIES = 4  # Number of retries for the LLM to fix its own JSON output or SQL errors (Total 5 attempts)
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

    async def _handle_exception(self, e, user_query=None, context=None):
        """
        Handle exceptions during SQL generation.
        """
        error_msg = f"Error: {e}"
        
        # Add context information if available
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_msg += f" [Context: {context_str}]"
        
        # Add user query if available
        if user_query:
            error_msg += f" [Query: {user_query}]"
        
        # Log the error
        print(error_msg, file=sys.stderr)
        
        # Return the error message for the caller
        return str(e)

    async def generate_sql_query(
        self,
        natural_language_question: str,
        schema_and_sample_data: Optional[Dict[str, Any]],
        insights_markdown_content: Optional[str],
        llm_client: Any,  # This will be provided by fastWorkflow
        row_limit_for_preview: int = 1
    ) -> Dict[str, Any]:
        """
        Generates an SQL query based on a natural language question, schema data, and insights.
        
        Args:
            natural_language_question: The user's question
            schema_and_sample_data: Dictionary containing schema and sample data for tables
            insights_markdown_content: String content of the cumulative insights markdown file
            llm_client: The LLM client provided by fastWorkflow
            row_limit_for_preview: Maximum number of rows to return in the preview
            
        Returns:
            A dictionary containing the SQL query, explanation, execution results, and user message
        """
        # Check if the client is available
        if not llm_client:
            return {
                "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
                "message_to_user": "Error: LLM client not available for SQL generation."
            }
            
        # Log the start of SQL generation
        _log_sql_generation_debug(self.db_name_identifier, "sql_generation_started", {
            "natural_language_question": natural_language_question,
            "has_schema_data": schema_and_sample_data is not None,
            "has_insights": insights_markdown_content is not None and len(insights_markdown_content.strip()) > 0
        })

        # --- RAG: Retrieve Few-Shot Examples ---
        few_shot_examples_str = "No similar approved queries found to use as examples."
        # Attempt to retrieve RAG examples, but proceed even if it fails.
        try:
            if vector_store_module and hasattr(vector_store_module, 'search_similar_nlqs'):
                # Using the hardcoded threshold from vector_store_module
                current_rag_threshold = getattr(vector_store_module, 'LITELLM_RAG_THRESHOLD', 0.75)

                # Log RAG retrieval start
                _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_started", {
                    "natural_language_question": natural_language_question,
                    "threshold": current_rag_threshold,
                    "k": 3
                })

                try:
                    # Call the search function and handle potential non-list returns
                    try:
                        similar_pairs = vector_store_module.search_similar_nlqs(
                            db_name_identifier=self.db_name_identifier,
                            query_nlq=natural_language_question,
                            k=3,  # Retrieve top 3 for few-shot prompting
                            threshold=current_rag_threshold
                        )
                    except Exception as e_call:
                        # Log specific error from the function call
                        _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_function_error", {
                            "error": str(e_call),
                            "error_type": type(e_call).__name__
                        })
                        similar_pairs = []
                    
                    # Ensure similar_pairs is a list and not None or an integer
                    if similar_pairs is None or not hasattr(similar_pairs, '__iter__') or isinstance(similar_pairs, (int, float, str)):
                        _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_type_error", {
                            "error": f"Expected list, got {type(similar_pairs).__name__}",
                            "value": str(similar_pairs)
                        })
                        similar_pairs = []
                    
                    # Log RAG retrieval results
                    _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_completed", {
                        "success": bool(similar_pairs),
                        "pairs_found": len(similar_pairs),
                        "threshold_used": current_rag_threshold
                    })
                    
                    if similar_pairs:
                        examples_parts = ["Here are some examples of approved natural language questions and their corresponding SQL queries for this database:\n"]
                        for i, pair in enumerate(similar_pairs):
                            examples_parts.append(f"Example {i+1}:")
                            examples_parts.append(f"  Natural Language Question: \"{pair['nlq']}\"")
                            examples_parts.append(f"  SQL Query: ```sql\n{pair['sql']}\n```")
                        few_shot_examples_str = "\n".join(examples_parts)
                except Exception as e_search:
                    _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_error", {
                        "error": str(e_search),
                        "error_type": type(e_search).__name__
                    })
        except Exception as e_rag:
            await self._handle_exception(e_rag, natural_language_question, {"context": "RAG few-shot example retrieval"})
            # Log RAG retrieval error
            _log_sql_generation_debug(self.db_name_identifier, "rag_retrieval_error", {
                "error": str(e_rag),
                "error_type": type(e_rag).__name__
            })
        # --- End RAG ---

        # --- HyDE: Retrieve Focused Schema Context ---
        hyde_context = ""
        table_names_from_hyde = []
        try:
            if hyde_module:
                # Log HyDE retrieval start
                _log_sql_generation_debug(self.db_name_identifier, "hyde_started", {
                    "natural_language_question": natural_language_question
                })
                
                hyde_context, table_names_from_hyde = await hyde_module.retrieve_hyde_context(
                    nlq=natural_language_question,
                    db_name_identifier=self.db_name_identifier,
                    llm_client=llm_client
                )
                
                # Log HyDE retrieval results
                _log_sql_generation_debug(self.db_name_identifier, "hyde_completed", {
                    "success": "Failed" not in hyde_context and "Error" not in hyde_context,
                    "context_length": len(hyde_context),
                    "table_names_found": table_names_from_hyde,
                    "context_snippet": hyde_context[:200] + "..." if len(hyde_context) > 200 else hyde_context
                })
        except Exception as e_hyde:
            await self._handle_exception(e_hyde, natural_language_question, {"context": "HyDE Context Retrieval"})
            hyde_context = "Failed to retrieve focused schema context via HyDE."
            # Log HyDE retrieval error
            _log_sql_generation_debug(self.db_name_identifier, "hyde_error", {
                "error": str(e_hyde),
                "error_type": type(e_hyde).__name__
            })
        # --- End HyDE ---

        # Prepare context strings for the prompt
        schema_context_str = "No schema or sample data provided."
        if schema_and_sample_data:
            try:
                # If HyDE context is available and not an error message, use it.
                # Otherwise, use a minimal context with just table names instead of the full schema
                if hyde_context and "Failed" not in hyde_context and "Error" not in hyde_context:
                    schema_context_str = hyde_context
                    _log_sql_generation_debug(self.db_name_identifier, "context_selection", {
                        "source": "hyde",
                        "context_length": len(hyde_context),
                        "table_names": table_names_from_hyde
                    })
                else:
                    # Instead of using full schema, use a truncated version or just table names
                    if table_names_from_hyde:
                        schema_context_str = f"Tables mentioned: {', '.join(table_names_from_hyde)}"
                    else:
                        # Try to extract some basic table information without using the full schema
                        table_names = list(schema_and_sample_data.keys())[:5]  # Limit to first 5 tables
                        schema_context_str = f"Available tables include: {', '.join(table_names)}"
                    
                    _log_sql_generation_debug(self.db_name_identifier, "context_selection", {
                        "source": "fallback",
                        "error": "HyDE failed to provide usable context",
                        "fallback_context": schema_context_str
                    })
            except TypeError as e:
                schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
                await self._handle_exception(e, natural_language_question, {"context": "Serializing schema data"})
                _log_sql_generation_debug(self.db_name_identifier, "context_selection_error", {
                    "error": str(e),
                    "error_type": type(e).__name__
                })

        insights_context_str = "No cumulative insights provided."
        if insights_markdown_content and insights_markdown_content.strip():
            insights_context_str = insights_markdown_content

        # --- Load Schema Graph and Find Join Path ---
        schema_graph = None
        join_path_str = "No deterministic join path could be constructed."
        if memory_module and table_names_from_hyde:
            try:
                # Log join path finding start
                _log_sql_generation_debug(self.db_name_identifier, "join_path_finding_started", {
                    "tables_for_join": table_names_from_hyde
                })
                
                schema_graph = memory_module.load_schema_graph(self.db_name_identifier)
                if schema_graph and join_path_finder:
                    join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
                    if join_clauses:
                        join_path_str = "\n".join(join_clauses)
                
                # Log join path finding results
                _log_sql_generation_debug(self.db_name_identifier, "join_path_finding_completed", {
                    "tables_for_join": table_names_from_hyde,
                    "join_path_found": join_path_str != "No deterministic join path could be constructed.",
                    "join_clauses": join_clauses if join_clauses else []
                })
            except Exception as e_join:
                await self._handle_exception(e_join, natural_language_question, {"context": "Join Path Finder"})
                join_path_str = "Error constructing join path."
                # Log join path finding error
                _log_sql_generation_debug(self.db_name_identifier, "join_path_finding_error", {
                    "error": str(e_join),
                    "error_type": type(e_join).__name__
                })
        # --- End Join Path Finding ---

        # Initial prompt for SQL generation
        current_prompt = (
            f"You are an expert PostgreSQL SQL writer. Based on the following database schema, sample data, "
            f"cumulative insights, and the natural language question, generate an appropriate SQL query "
            f"(must start with SELECT) and a brief explanation.\n\n"
            f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the query):\n```\n{schema_context_str}\n```\n\n"
            f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query):\n```\n{join_path_str}\n```\n\n"
            f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
            f"FEW-SHOT EXAMPLES (Use these to guide your SQL generation if relevant):\n{few_shot_examples_str}\n\n"
            f"NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
            f"IMPORTANT: Your response MUST be a single, valid JSON object. Do not include any other text, explanations, or markdown formatting outside of the JSON structure.\n"
            f"The JSON object must conform to this exact structure: "
            f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}"
        )
        
        # Use a rough estimate for token count instead of importing token_utils
        prompt_token_count = len(current_prompt) // 4  # Rough estimate
            
        # Log prompt preparation
        _log_sql_generation_debug(self.db_name_identifier, "llm_prompt_prepared", {
            "token_count": prompt_token_count,
            "prompt_length": len(current_prompt),
            "schema_context_length": len(schema_context_str),
            "join_path_length": len(join_path_str),
            "insights_length": len(insights_context_str),
            "few_shot_examples_length": len(few_shot_examples_str)
        })

        llm_response_text = ""  # Will store the final text part of LLM response
        parsed_response = None
        last_error_feedback_to_llm = ""  # This will be a user-role message for LiteLLM

        # Initial messages list for LiteLLM
        messages_for_llm = []
        if hasattr(llm_client, 'conversation_history'):
            messages_for_llm = llm_client.conversation_history[:]

        for attempt in range(self.MAX_SQL_GEN_RETRIES + 1):
            user_message_content = ""
            if attempt > 0 and last_error_feedback_to_llm:  # This implies a retry
                user_message_content = (
                    f"{last_error_feedback_to_llm}\n\n"  # last_error_feedback_to_llm is the *previous* error message from assistant
                    f"Based on the previous error, please try again.\n"
                    f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                    f"DETERMINISTIC JOIN PATH:\n```\n{join_path_str}\n```\n\n"
                    f"CUMULATIVE INSIGHTS:\n```markdown\n{insights_context_str}\n```\n\n"
                    f"FEW-SHOT EXAMPLES:\n{few_shot_examples_str}\n\n"  # Added few-shot examples to retry prompt
                    f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                    f"Respond ONLY with a single JSON object matching the structure: "
                    f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                    f"Ensure the SQL query strictly starts with 'SELECT'."
                )
            else:  # First attempt
                user_message_content = current_prompt  # current_prompt is the initial full prompt

            # Add the current user prompt to messages for this specific call
            # We use a temporary list for the call to avoid polluting main history with retries until success
            current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]

            try:
                # Log LLM request
                _log_sql_generation_debug(self.db_name_identifier, "llm_request_sent", {
                    "attempt": attempt,
                    "is_retry": attempt > 0
                })
                
                # Send the message to the LLM
                if hasattr(llm_client, '_send_message_to_llm'):
                    # Use the client's method if available
                    llm_response_obj = await llm_client._send_message_to_llm(current_call_messages, natural_language_question, 0, source='sql_generation')
                    llm_response_text, tool_calls_made = await llm_client._process_llm_response(llm_response_obj)
                else:
                    # Fallback to a simpler approach
                    llm_response = await llm_client.chat(current_call_messages)
                    llm_response_text = llm_response.choices[0].message.content
                    tool_calls_made = False

                # Log LLM response received
                _log_sql_generation_debug(self.db_name_identifier, "llm_response_received", {
                    "attempt": attempt,
                    "response_length": len(llm_response_text),
                    "tool_calls_made": tool_calls_made
                })
                
                if tool_calls_made:
                    last_error_feedback_to_llm = "Your response included an unexpected tool call. Please provide the JSON response directly."
                    _log_sql_generation_debug(self.db_name_identifier, "llm_response_error", {
                        "attempt": attempt,
                        "error": "Tool call made instead of JSON response",
                        "will_retry": attempt < self.MAX_SQL_GEN_RETRIES
                    })
                    if attempt < self.MAX_SQL_GEN_RETRIES:
                        continue
                    else:
                        raise Exception("LLM attempted tool call instead of providing JSON for SQL generation.")

                json_from_response = self._extract_json_from_response(llm_response_text)

                if not json_from_response:
                    last_error_feedback_to_llm = "Your response did not contain a valid JSON object. Please provide the JSON output as instructed."
                    _log_sql_generation_debug(self.db_name_identifier, "llm_response_error", {
                        "attempt": attempt,
                        "error": "No JSON found in response",
                        "will_retry": attempt < self.MAX_SQL_GEN_RETRIES,
                        "response_snippet": llm_response_text[:200] + "..." if len(llm_response_text) > 200 else llm_response_text
                    })
                    if attempt < self.MAX_SQL_GEN_RETRIES:
                        continue
                    else:
                        raise Exception("LLM failed to provide a JSON response for SQL generation.")

                try:
                    parsed_response = SQLGenerationResponse.model_validate_json(json_from_response)
                    _log_sql_generation_debug(self.db_name_identifier, "json_validation_success", {
                        "attempt": attempt,
                        "has_sql_query": bool(parsed_response.sql_query),
                        "has_explanation": bool(parsed_response.explanation)
                    })
                except ValidationError as ve:
                    last_error_feedback_to_llm = f"Your response contained invalid JSON: {str(ve)}. Please provide a valid JSON object."
                    _log_sql_generation_debug(self.db_name_identifier, "json_validation_error", {
                        "attempt": attempt,
                        "error": str(ve),
                        "will_retry": attempt < self.MAX_SQL_GEN_RETRIES,
                        "json_snippet": json_from_response[:200] + "..." if len(json_from_response) > 200 else json_from_response
                    })
                    if attempt < self.MAX_SQL_GEN_RETRIES:
                        continue
                    else:
                        raise Exception(f"LLM failed to provide valid JSON after retries: {str(ve)}")
                
                if not parsed_response.sql_query:
                    error_detail = parsed_response.error_message or "LLM did not provide an SQL query in the 'sql_query' field."
                    last_error_feedback_to_llm = (
                        f"Your previous attempt did not produce an SQL query in the 'sql_query' field. LLM message: '{error_detail}'. "
                        f"Ensure 'sql_query' field is populated with a valid SQL string."
                    )
                    _log_sql_generation_debug(self.db_name_identifier, "sql_query_missing", {
                        "attempt": attempt,
                        "error": error_detail,
                        "will_retry": attempt < self.MAX_SQL_GEN_RETRIES
                    })
                    if attempt < self.MAX_SQL_GEN_RETRIES:
                        continue
                    else:
                        raise Exception(f"LLM failed to provide SQL query after retries. Last message: {error_detail}")

                if not parsed_response.sql_query.strip().upper().startswith("SELECT"):
                    last_error_feedback_to_llm = (
                        f"Your generated SQL query was: ```sql\n{parsed_response.sql_query}\n```\n"
                        f"This query does not start with SELECT, which is a requirement. Please regenerate a valid SELECT query."
                    )
                    _log_sql_generation_debug(self.db_name_identifier, "non_select_query", {
                        "attempt": attempt,
                        "query": parsed_response.sql_query,
                        "will_retry": attempt < self.MAX_SQL_GEN_RETRIES
                    })
                    if attempt < self.MAX_SQL_GEN_RETRIES:
                        parsed_response = None  # Invalidate this response
                        continue
                    else:
                        raise Exception("LLM failed to generate a SELECT query after retries.")
                
                # Log successful SQL generation
                _log_sql_generation_debug(self.db_name_identifier, "sql_generation_success", {
                    "attempt": attempt,
                    "sql_query": parsed_response.sql_query,
                    "explanation_length": len(parsed_response.explanation) if parsed_response.explanation else 0
                })
                
                break

            except Exception as e:
                context = {
                    "context": "SQL generation loop",
                    "attempt": attempt + 1,
                    "llm_response_text": llm_response_text,
                    "db_name_identifier": self.db_name_identifier
                }
                user_message = await self._handle_exception(e, natural_language_question, context)
                last_error_feedback_to_llm = f"A validation error occurred: {user_message}. Please try again."
                _log_sql_generation_debug(self.db_name_identifier, "sql_generation_exception", {
                    "attempt": attempt,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "will_retry": attempt < self.MAX_SQL_GEN_RETRIES
                })
                if attempt == self.MAX_SQL_GEN_RETRIES:
                    return {
                        "sql_query": None, "explanation": None, "execution_result": None, "execution_error": user_message,
                        "message_to_user": "Failed to generate SQL query after all attempts."  # Generic message for the main part
                    }
        # End of retry loop

        if not parsed_response or not parsed_response.sql_query:
            return {
                "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
                "message_to_user": "Failed to generate SQL query after all attempts."
            }

        sql_to_execute = parsed_response.sql_query
        explanation = parsed_response.explanation
        execution_result = None
        execution_error = None
        
        # Execute SQL with retry logic for execution errors
        original_sql = sql_to_execute
        original_explanation = explanation
        
        for exec_attempt in range(self.MAX_SQL_EXECUTION_RETRIES + 1):
            try:
                
                # Execute the SQL query using the client's session if available
                if hasattr(llm_client, 'session') and llm_client.session:
                    # Use the session directly from the LLM client
                    try:
                        exec_result_obj = await llm_client.session.call_tool(
                            "execute_postgres_query",
                            {"query": sql_to_execute, "row_limit": row_limit_for_preview}
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
                                    tables_in_query = self._extract_tables_from_query(sql_to_execute)
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
                                                await self._handle_exception(e_desc, natural_language_question, {"context": f"Describing table {table_name} in retry loop"})
                                        detailed_schema_context = "\n".join(table_schemas)
                                except Exception as e_extract:
                                    await self._handle_exception(e_extract, natural_language_question, {"context": "Extracting tables from query in retry loop"})
                                # --- End Intelligent Retry ---

                                fix_user_message_content = (
                                    f"The previously generated SQL query resulted in an execution error.\n"
                                    f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                                    f"SQL QUERY WITH ERROR:\n```sql\n{sql_to_execute}\n```\n\n"
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
                                        fix_llm_response_obj = await llm_client._send_message_to_llm(fix_call_messages, natural_language_question, 0, source='sql_generation')
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
                                            sql_to_execute = fixed_response.sql_query
                                            explanation = fixed_response.explanation
                                            execution_error = None 
                                            continue
                                    except ValidationError:
                                        continue
                                except Exception as fix_e:
                                    await self._handle_exception(fix_e, natural_language_question, {"context": "LLM SQL fix attempt", "attempt": exec_attempt + 1})
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
                execution_error = await self._handle_exception(e, natural_language_question, {"context": "SQL execution loop", "attempt": exec_attempt + 1, "sql_to_execute": sql_to_execute})
                if exec_attempt < self.MAX_SQL_EXECUTION_RETRIES:
                    continue
        
        if execution_error and sql_to_execute != original_sql:
            # Revert to original SQL for display if all fix attempts failed
            sql_to_execute = original_sql
            explanation = original_explanation

        user_message = f"Generated SQL:\n```sql\n{sql_to_execute}\n```\n\nExplanation:\n{explanation}\n\n"
        
        if execution_error:
            user_message += f"Execution Error: {execution_error}\n\n"
        elif execution_result is not None:
            # Format the execution result for display
            if isinstance(execution_result, dict) and execution_result.get("status") == "success":
                data = execution_result.get("data")
                if data and isinstance(data, list) and len(data) > 0:
                    user_message += f"Execution Result (first {min(len(data), row_limit_for_preview)} rows):\n"
                    for i, row in enumerate(data[:row_limit_for_preview]):
                        user_message += f"{row}\n"
                    if len(data) > row_limit_for_preview:
                        user_message += f"... and {len(data) - row_limit_for_preview} more rows\n"
                else:
                    user_message += "Query executed successfully, but no rows were returned.\n"
            else:
                user_message += f"Execution Result: {execution_result}\n"

        user_message += "\nIf this is correct, use '/approved'. If not, use '/feedback Your feedback text'."
        
        return {
            "sql_query": sql_to_execute,
            "explanation": explanation,
            "execution_result": execution_result,
            "execution_error": execution_error,
            "message_to_user": user_message
        }
