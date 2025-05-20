import json
from typing import TYPE_CHECKING, Dict, Any, Optional

from pydantic import ValidationError
from pydantic_models import SQLGenerationResponse # Assuming SQLErrorFeedbackResponse is not used in this simplified version

if TYPE_CHECKING:
    from postgres_copilot_chat import GeminiMcpClient # To avoid circular import

MAX_SQL_GEN_RETRIES = 4 # Number of retries for the LLM to fix its own JSON output or SQL errors (Total 5 attempts)
MAX_SQL_EXECUTION_RETRIES = 5 # Number of retries for the LLM to fix SQL execution errors

async def generate_sql_query(
    client: 'GeminiMcpClient', 
    natural_language_question: str,
    schema_and_sample_data: Optional[Dict[str, Any]], # Combined DDL and sample data
    insights_markdown_content: Optional[str] # Content of summarized_insights.md
) -> Dict[str, Any]:
    """
    Generates an SQL query based on a natural language question, schema data, and insights.
    Validates the LLM's response and executes the query for verification.

    Args:
        client: The GeminiMcpClient instance.
        natural_language_question: The user's question.
        schema_and_sample_data: Dictionary containing schema (DDL-like) and sample data for tables.
        insights_markdown_content: String content of the cumulative insights markdown file.

    Returns:
        A dictionary containing the SQL query, explanation, execution results, and user message.
    """
    if not client.chat or not client.session:
        return {
            "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
            "message_to_user": "Error: LLM chat or MCP session not available for SQL generation."
        }

    # Prepare context strings for the prompt
    schema_context_str = "No schema or sample data provided."
    if schema_and_sample_data:
        try:
            schema_context_str = json.dumps(schema_and_sample_data, indent=2)
        except TypeError as e:
            schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
            print(f"Warning: Could not serialize schema_and_sample_data to JSON: {e}")


    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content

    # Initial prompt for SQL generation
    current_prompt = (
        f"You are an expert PostgreSQL SQL writer. Based on the following database schema, sample data, "
        f"cumulative insights, and the natural language question, generate an appropriate SQL query "
        f"(must start with SELECT) and a brief explanation.\n\n"
        f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
        f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
        f"NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
        f"Respond ONLY with a single JSON object matching this structure: "
        f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
        f"Ensure the SQL query strictly starts with 'SELECT'."
    )

    llm_response_text = ""
    parsed_response: Optional[SQLGenerationResponse] = None
    last_error_feedback_to_llm = ""

    for attempt in range(MAX_SQL_GEN_RETRIES + 1): # Initial attempt + retries
        print(f"Attempting SQL generation (Attempt {attempt + 1}/{MAX_SQL_GEN_RETRIES + 1})...")
        if attempt > 0 and last_error_feedback_to_llm: # Modify prompt for retries based on specific errors
             current_prompt = (
                f"{last_error_feedback_to_llm}\n\n"
                f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                f"CUMULATIVE INSIGHTS:\n```markdown\n{insights_context_str}\n```\n\n"
                f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                f"Please try again. Respond ONLY with a single JSON object matching the structure: "
                f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                f"Ensure the SQL query strictly starts with 'SELECT'."
            )

        try:
            response = await client.chat.send_message_async(current_prompt)
            
            current_llm_response_text = ""
            if response.parts:
                for part_item in response.parts: # Renamed to avoid conflict
                    if hasattr(part_item, 'text') and part_item.text:
                        current_llm_response_text += part_item.text
            else:
                current_llm_response_text = str(response)
            
            llm_response_text = current_llm_response_text
            print(f"LLM raw response for SQL gen (Attempt {attempt + 1}): {llm_response_text}")

            cleaned_llm_response_text = llm_response_text.strip()
            if cleaned_llm_response_text.startswith("```json"):
                cleaned_llm_response_text = cleaned_llm_response_text[7:]
            if cleaned_llm_response_text.endswith("```"):
                cleaned_llm_response_text = cleaned_llm_response_text[:-3]
            cleaned_llm_response_text = cleaned_llm_response_text.strip()

            parsed_response = SQLGenerationResponse.model_validate_json(cleaned_llm_response_text)
            
            if not parsed_response.sql_query:
                error_detail = parsed_response.error_message or "LLM did not provide an SQL query."
                last_error_feedback_to_llm = (
                    f"Your previous attempt did not produce an SQL query. LLM message: '{error_detail}'. "
                    f"Ensure 'sql_query' field is populated."
                )
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception(f"LLM failed to provide SQL query after retries. Last message: {error_detail}")

            if not parsed_response.sql_query.strip().upper().startswith("SELECT"):
                last_error_feedback_to_llm = (
                    f"Your generated SQL query was: ```sql\n{parsed_response.sql_query}\n```\n"
                    f"This query does not start with SELECT, which is a requirement. Please regenerate."
                )
                if attempt < MAX_SQL_GEN_RETRIES:
                    parsed_response = None # Invalidate this response
                    continue
                else: raise Exception("LLM failed to generate a SELECT query after retries.")
            
            print("SQL Generation successful and format validated.")
            break # Exit loop on successful parse and basic validation

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"LLM response validation error for SQL (Attempt {attempt + 1}): {e}")
            last_error_feedback_to_llm = (
                f"Your previous response was not in the correct JSON format or had validation issues. Error: {e}\n"
                f"Original incorrect response snippet: {llm_response_text[:200]}"
            )
            if attempt == MAX_SQL_GEN_RETRIES:
                return {
                    "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
                    "message_to_user": f"Failed to get a valid SQL response from LLM. Last error: {e}. Last LLM response: {llm_response_text}"
                }
        except Exception as e: # Catch other errors, including those raised for no SQL or non-SELECT
            print(f"Error during SQL generation attempt {attempt + 1}: {e}")
            last_error_feedback_to_llm = f"An error occurred: {e}. Please try again."
            if attempt == MAX_SQL_GEN_RETRIES:
                return {
                    "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
                    "message_to_user": f"Failed to generate SQL after multiple attempts. Last error: {e}"
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
    
    for exec_attempt in range(MAX_SQL_EXECUTION_RETRIES + 1):  # Initial attempt + retries
        print(f"Executing SQL (Attempt {exec_attempt + 1}/{MAX_SQL_EXECUTION_RETRIES + 1}): {sql_to_execute}")
        
        try:
            exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": sql_to_execute})
            raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj)
            
            if isinstance(raw_exec_output, str) and "Error:" in raw_exec_output:
                execution_error = raw_exec_output
                print(f"Execution Error (Attempt {exec_attempt + 1}): {execution_error}")
                
                # If we have more attempts, try to fix the SQL
                if exec_attempt < MAX_SQL_EXECUTION_RETRIES:
                    # Create a prompt for the LLM to fix the SQL
                    fix_prompt = (
                        f"You are an expert PostgreSQL SQL writer. You need to fix an SQL query that produced an error.\n\n"
                        f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                        f"NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                        f"INCORRECT SQL QUERY:\n```sql\n{sql_to_execute}\n```\n\n"
                        f"ERROR MESSAGE:\n{execution_error}\n\n"
                        f"Please fix the SQL query and provide a brief explanation of what was wrong and how you fixed it.\n"
                        f"Respond ONLY with a single JSON object matching this structure: "
                        f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                        f"Ensure the SQL query strictly starts with 'SELECT'."
                    )
                    
                    try:
                        fix_response = await client.chat.send_message_async(fix_prompt)
                        fix_text = "".join(part.text for part in fix_response.parts if hasattr(part, 'text') and part.text).strip()
                        
                        # Clean up the response
                        if fix_text.startswith("```json"):
                            fix_text = fix_text[7:]
                        if fix_text.endswith("```"):
                            fix_text = fix_text[:-3]
                        fix_text = fix_text.strip()
                        
                        # Parse the fixed SQL
                        fixed_response = SQLGenerationResponse.model_validate_json(fix_text)
                        
                        if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                            # Update the SQL and explanation for the next attempt
                            sql_to_execute = fixed_response.sql_query
                            explanation = fixed_response.explanation
                            execution_error = None  # Reset error for next attempt
                            print(f"SQL fixed for next attempt: {sql_to_execute}")
                            continue  # Try again with the fixed SQL
                        else:
                            print("LLM provided invalid SQL fix. Continuing with next attempt...")
                    except Exception as fix_e:
                        print(f"Error while trying to fix SQL (Attempt {exec_attempt + 1}): {fix_e}")
                        # Continue to next attempt with same SQL
            else:
                # Success! No error
                execution_result = raw_exec_output
                execution_error = None
                print(f"Execution Successful. Result type: {type(execution_result)}")
                break  # Exit the retry loop on success
                
        except Exception as e:
            print(f"Exception during execute_postgres_query tool call (Attempt {exec_attempt + 1}): {e}")
            execution_error = f"Exception while trying to execute generated SQL: {e}"
            
            # Continue to next attempt if we have retries left
            if exec_attempt < MAX_SQL_EXECUTION_RETRIES:
                continue
    
    # If we've exhausted all attempts and still have an error, revert to the original SQL for display
    if execution_error and sql_to_execute != original_sql:
        print("All SQL fix attempts failed. Reverting to original SQL for display.")
        sql_to_execute = original_sql
        explanation = original_explanation

    user_message = f"Generated SQL:\n```sql\n{sql_to_execute}\n```\n\nExplanation:\n{explanation}\n\n"
    # Execution results/errors are no longer appended to the user_message.
    # They are still printed to the console during the execution loop above.
    if execution_error:
        # This error was already printed to console during the execution attempt.
        # We just note it here for the return dict, but not for user_message.
        print(f"SQL Execution Error (not shown to user): {execution_error}")
    else:
        # This result was already printed to console.
        print(f"SQL Execution Successful (result not shown to user).")

    user_message += "If this is correct, use '/approved'. If not, use '/feedback Your feedback text'."

    return {
        "sql_query": sql_to_execute,
        "explanation": explanation,
        "execution_result": execution_result,
        "execution_error": execution_error,
        "message_to_user": user_message
    }
