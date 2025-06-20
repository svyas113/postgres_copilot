import json
import re # Added for extracting table name
from typing import TYPE_CHECKING, Dict, Any, Optional

from .pydantic_models import SQLGenerationResponse # Changed to relative
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from .postgres_copilot_chat import LiteLLMMcpClient # Changed to relative

# Pydantic model for LLM's decision on how to handle navigation
class NavigationAction(BaseModel):
    action_type: str = Field(description="Type of action to take: 'execute_sql', 'get_schema', 'answer_directly', 'clarify_question'")
    sql_query_to_execute: Optional[str] = Field(None, description="SQL query to execute if action_type is 'execute_sql'.")
    explanation_for_user: Optional[str] = Field(None, description="Explanation or direct answer for the user.")
    clarification_question_for_user: Optional[str] = Field(None, description="A question to ask the user for clarification.")

async def handle_navigation_query(
    client: 'LiteLLMMcpClient', 
    user_query: str,
    current_db_name: str, 
    insights_markdown_content: Optional[str],
    raw_schema_and_sample_data: Optional[Dict[str, Any]] 
) -> str:
    """
    Handles natural language queries for database navigation and exploration using LiteLLM.
    The LLM decides whether to execute a query, fetch schema, or answer directly.
    """
    if not client.session: # For LiteLLM client, session is the key check for MCP
        return "Error: MCP session not available for database navigation."

    schema_context_str = "No schema or sample data available for direct reference in this navigation prompt, but you can use tools to fetch it."
    if raw_schema_and_sample_data:
        try:
            table_names = list(raw_schema_and_sample_data.keys())
            schema_context_str = (
                f"The connected database ('{current_db_name}') has the following tables (details can be fetched using tools): {', '.join(table_names)}. "
                f"Full DDL and sample data were loaded at initialization."
            )
        except Exception as e:
            print(f"Error processing schema for navigation prompt context: {e}")
            schema_context_str = "Schema and sample data were loaded at initialization, but cannot be summarized here. Use tools to fetch details."


    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content
    
    navigation_action_schema = NavigationAction.model_json_schema()

    prompt = (
        f"You are a database exploration assistant for a PostgreSQL database named '{current_db_name}'. "
        f"The user wants to navigate or understand the database with the following query: \"{user_query}\"\n\n"
        f"Available context:\n"
        f"1. Schema Overview: {schema_context_str}\n"
        f"2. Cumulative Insights (from previous interactions):\n```markdown\n{insights_context_str}\n```\n\n"
        f"Your task is to decide the best course of action. You have access to the following tools on the MCP server:\n"
        f"- `get_schema_and_sample_data()`: Fetches DDL-like statements and 5 sample rows for all tables. Use this if the user asks for schema details, table structures, or general overview.\n"
        f"- `execute_postgres_query(query: str)`: Executes any valid PostgreSQL query. Use this for specific data requests (e.g., 'show me 10 customers', 'count orders').\n\n"
        f"Based on the user's query, decide on an action. Respond ONLY with a single JSON object. "
        f"The JSON object MUST conform to the following structure (provide only the JSON, no other text before or after it, especially no ```json markdown backticks around the JSON output itself):\n"
        f"- `action_type` (string, required): Choose one of 'execute_sql', 'get_schema', 'answer_directly', 'clarify_question'.\n"
        f"- `sql_query_to_execute` (string, optional): If `action_type` is 'execute_sql', provide the SQL query here. It MUST start with SELECT.\n"
        f"- `explanation_for_user` (string, optional): If `action_type` is 'answer_directly', provide the answer. If 'execute_sql' or 'get_schema', briefly explain what you're about to do.\n"
        f"- `clarification_question_for_user` (string, optional): If `action_type` is 'clarify_question', provide the question to ask the user.\n\n"
        f"Example of a valid JSON response for 'execute_sql':\n"
        f"{{\n"
        f"  \"action_type\": \"execute_sql\",\n"
        f"  \"sql_query_to_execute\": \"SELECT * FROM your_table WHERE id = 123;\",\n"
        f"  \"explanation_for_user\": \"I will run a query to select all columns from your_table for the specified ID.\"\n"
        f"}}\n"
        f"Example of a valid JSON response for 'clarify_question':\n"
        f"{{\n"
        f"  \"action_type\": \"clarify_question\",\n"
        f"  \"clarification_question_for_user\": \"Which specific columns are you interested in from the customer table?\"\n"
        f"}}\n"
        f"Guidelines for `NavigationAction` fields:\n"
        f"- `action_type`: Choose one of 'execute_sql', 'get_schema', 'answer_directly', 'clarify_question'.\n"
        f"  - 'execute_sql': If the user's query can be directly translated into a SELECT SQL query to fetch data.\n"
        f"  - 'get_schema': If the user asks about table structure, columns, or wants a general database overview.\n"
        f"  - 'answer_directly': If the query is a general question that can be answered from the provided insights or general knowledge without hitting the database (less common for navigation).\n"
        f"  - 'clarify_question': If the user's query is too vague or ambiguous to proceed.\n"
        f"- `sql_query_to_execute`: If `action_type` is 'execute_sql', provide the SQL query here. It MUST start with SELECT.\n"
        f"- `explanation_for_user`: If `action_type` is 'answer_directly', provide the answer. If 'execute_sql' or 'get_schema', briefly explain what you're about to do.\n"
        f"- `clarification_question_for_user`: If `action_type` is 'clarify_question', provide the question to ask the user.\n"
    )

    MAX_NAV_RETRIES = 1 
    llm_response_text = "" 
    last_error_for_retry_prompt = ""
    messages_for_llm = client.conversation_history[:]

    for attempt in range(MAX_NAV_RETRIES + 1):
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to decide the navigation action again, adhering to the JSON schema.\n"
                f"Original user query: \"{user_query}\". Database: '{current_db_name}'.\n"
                f"Respond ONLY with the `NavigationAction` JSON object."
            )

        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]
        
        try:
            llm_response_obj = await client._send_message_to_llm(current_call_messages)
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON decision directly."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM attempted tool call instead of providing JSON decision for navigation."
                continue

            if llm_response_text.startswith("```json"): llm_response_text = llm_response_text[7:]
            if llm_response_text.endswith("```"): llm_response_text = llm_response_text[:-3]
            llm_response_text = llm_response_text.strip()

            if not llm_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON decision."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM provided an empty response for navigation decision."
                continue
            
            action_decision = NavigationAction.model_validate_json(llm_response_text)

            if action_decision.action_type == 'execute_sql':
                # ... (execute_sql logic remains the same)
                if not action_decision.sql_query_to_execute or not action_decision.sql_query_to_execute.strip().upper().startswith("SELECT"):
                    return "LLM decided to execute SQL but did not provide a valid SELECT query. Please try rephrasing."
                user_feedback = (
                    f"{action_decision.explanation_for_user or 'Executing SQL as per your request.'}\n"
                    f"Executing SQL:\n```sql\n{action_decision.sql_query_to_execute}\n```\n"
                )
                try:
                    exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": action_decision.sql_query_to_execute})
                    raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj)
                except Exception as e_tool_call:
                    return f"{user_feedback}\nError during tool execution: {e_tool_call}"
                if isinstance(raw_exec_output, str):
                    if "Error:" in raw_exec_output: return f"{user_feedback}\nExecution Error: {raw_exec_output}"
                    elif raw_exec_output == "meta=None content=[] isError=False":
                        query_lower = action_decision.sql_query_to_execute.lower()
                        if "information_schema.views" in query_lower or "information_schema.routines" in query_lower:
                            return f"{user_feedback}\nExecution Result: Query executed successfully. No user-defined views or functions found."
                        return f"{user_feedback}\nExecution Result: Query executed successfully. No results found."
                    else: return f"{user_feedback}\nExecution Result:\n{raw_exec_output}"
                elif isinstance(raw_exec_output, dict):
                    if raw_exec_output.get("error"): return f"{user_feedback}\nExecution Error: {raw_exec_output['error']}"
                    elif "data" in raw_exec_output:
                        data = raw_exec_output["data"]
                        if not data and isinstance(data, list):
                            query_lower = action_decision.sql_query_to_execute.lower()
                            if "information_schema.views" in query_lower or "information_schema.routines" in query_lower:
                                return f"{user_feedback}\nExecution Result: Query executed successfully. No user-defined views or functions found."
                            return f"{user_feedback}\nExecution Result: Query executed successfully. No results found."
                        else:
                            result_str = json.dumps(data, indent=2, default=str)
                            return f"{user_feedback}\nExecution Result:\n```json\n{result_str}\n```"
                    else: 
                        result_str = json.dumps(raw_exec_output, indent=2, default=str)
                        return f"{user_feedback}\nExecution Result (unexpected dictionary format):\n```json\n{result_str}\n```"
                elif isinstance(raw_exec_output, list):
                    if not raw_exec_output:
                        query_lower = action_decision.sql_query_to_execute.lower()
                        if "information_schema.views" in query_lower or "information_schema.routines" in query_lower:
                            return f"{user_feedback}\nExecution Result: Query executed successfully. No user-defined views or functions found."
                        return f"{user_feedback}\nExecution Result: Query executed successfully. No results found."
                    else:
                        result_str = json.dumps(raw_exec_output, indent=2, default=str)
                        return f"{user_feedback}\nExecution Result:\n```json\n{result_str}\n```"
                else: 
                    return f"{user_feedback}\nExecution Result (unknown data type: {type(raw_exec_output)}):\n{str(raw_exec_output)}"


            elif action_decision.action_type == 'get_schema':
                user_feedback_from_llm = action_decision.explanation_for_user or "Processing your schema request..."
                current_schema_data_to_use = None
                user_feedback_display = user_feedback_from_llm # Default to LLM's explanation

                if raw_schema_and_sample_data and isinstance(raw_schema_and_sample_data, dict) and raw_schema_and_sample_data:
                    current_schema_data_to_use = raw_schema_and_sample_data
                    user_feedback_display = "Accessing available schema information to answer your query..."
                else:
                    try:
                        schema_result_obj = await client.session.call_tool("get_schema_and_sample_data", {})
                        tool_output = client._extract_mcp_tool_call_output(schema_result_obj)
                        
                        if isinstance(tool_output, str) and "Error:" in tool_output:
                            return f"{user_feedback_display}\nError fetching schema via tool: {tool_output}"
                        
                        if isinstance(tool_output, dict) and tool_output:
                            current_schema_data_to_use = tool_output
                        # If tool_output is not a dict or empty, current_schema_data_to_use remains None
                        
                    except Exception as tool_e:
                        return f"{user_feedback_display}\nError calling get_schema_and_sample_data tool: {tool_e}"

                if current_schema_data_to_use:
                    target_table_name = None
                    match_table_word = re.search(r"table\s+([a-zA-Z_][a-zA-Z0-9_]*)", user_query, re.IGNORECASE)
                    if match_table_word: target_table_name = match_table_word.group(1)
                    else:
                        match_word_table = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+table", user_query, re.IGNORECASE)
                        if match_word_table: target_table_name = match_word_table.group(1)
                    
                    schema_to_explain_final = current_schema_data_to_use 
                    if target_table_name and target_table_name in current_schema_data_to_use:
                        schema_to_explain_final = {target_table_name: current_schema_data_to_use[target_table_name]}

                    schema_for_llm_str = json.dumps(schema_to_explain_final, indent=2)
                    
                    explanation_prompt_messages = client.conversation_history + [
                        {"role": "user",
                         "content": f"The user originally asked: \"{user_query}\"\n"
                                    f"You previously decided that schema information was necessary to answer this. "
                                    f"The following schema details (and possibly sample data) were retrieved/accessed for {'table ' + target_table_name if target_table_name and schema_to_explain_final.get(target_table_name) else 'the database'}:\n"
                                    f"```json\n{schema_for_llm_str}\n```\n"
                                    f"Your task now is to use this information to provide a concise, natural language answer to the user's original question: \"{user_query}\".\n"
                                    f"IMPORTANT: Your response should be **plain text only**. Do NOT output JSON. Do NOT use markdown code blocks like ```json. "
                                    f"Extract the relevant information from the schema and explain it clearly. "
                                    f"For example, if the user asked about columns in a table, list the column names and perhaps their types. If they asked about what a table contains, describe its purpose based on its columns and sample data (if available).\n"
                                    f"Focus on being helpful and directly addressing the query. Avoid technical jargon where possible, or explain it if necessary. "
                                    f"Do not mention the process of fetching/accessing data or this internal step. Just provide the answer as a natural language paragraph or bullet points."
                        }
                    ]
                    
                    try:
                        explanation_response_obj = await client._send_message_to_llm(explanation_prompt_messages)
                        raw_explanation_text, _ = await client._process_llm_response(explanation_response_obj)

                        final_natural_language_answer = raw_explanation_text # Default to raw text

                        if raw_explanation_text: # Ensure it's not None or empty
                            try:
                                # Attempt to parse if it's JSON that might contain the actual explanation
                                data = json.loads(raw_explanation_text)
                                if isinstance(data, dict):
                                    if "explanation_for_user" in data and isinstance(data["explanation_for_user"], str):
                                        final_natural_language_answer = data["explanation_for_user"]
                                    elif "answer" in data and isinstance(data["answer"], str):
                                        final_natural_language_answer = data["answer"]
                                    elif "content" in data and isinstance(data["content"], str): # Another common key for text
                                        final_natural_language_answer = data["content"]
                                    # If it's a dict but doesn't have these keys, final_natural_language_answer remains raw_explanation_text
                            except (json.JSONDecodeError, TypeError):
                                # Not JSON, or not a string initially, so final_natural_language_answer (raw_explanation_text) is used.
                                pass
                        
                        if not final_natural_language_answer: # Fallback if everything resulted in None/empty
                            final_natural_language_answer = "I found some schema information, but I'm having trouble explaining it in plain text right now."

                        return f"{user_feedback_display}\n{final_natural_language_answer}"
                    except Exception as e_explain:
                        if isinstance(current_schema_data_to_use, dict):
                            summary_parts = [f"{user_feedback_display}\nI accessed the schema information but encountered an issue generating a full natural language explanation (Error: {e_explain}). Here's a summary of the retrieved data:"]
                            for table, details in current_schema_data_to_use.items():
                                summary_parts.append(f"\nTable: {table}")
                                if isinstance(details, dict):
                                    if "ddl" in details: summary_parts.append(f"  Structure (DDL snippet): {details['ddl'][:200]}{'...' if len(details['ddl']) > 200 else ''}")
                                    if "sample_data" in details and details["sample_data"]: summary_parts.append(f"  Sample Data (first row if available): {details['sample_data'][0] if details['sample_data'] else 'N/A'}")
                            return "\n".join(summary_parts)
                        else: 
                            return f"{user_feedback_display}\nI accessed the schema, but had trouble explaining it (Error: {e_explain}). The raw data is: {str(current_schema_data_to_use)[:500]}..."
                else:
                    return f"{user_feedback_from_llm}\nI attempted to retrieve the schema information, but it appears to be unavailable or not in the expected format. I cannot provide a detailed summary based on this."

            elif action_decision.action_type == 'answer_directly':
                return action_decision.explanation_for_user or "I'm not sure how to answer that directly without more specific database interaction."

            elif action_decision.action_type == 'clarify_question':
                return action_decision.clarification_question_for_user or "Could you please clarify your request?"
            
            else: 
                return f"LLM chose an unknown or invalid action type: {action_decision.action_type}. Please try rephrasing."

        except (ValidationError, json.JSONDecodeError) as e:
            error_msg = f"LLM response validation error for navigation (Attempt {attempt + 1}): {e}"
            last_error_for_retry_prompt = f"{error_msg}\nProblematic response: {llm_response_text[:200]}"
            if attempt == MAX_NAV_RETRIES:
                return f"Error processing navigation request: Could not understand the desired action. Last LLM response: {llm_response_text}"
        except Exception as e:
            error_msg = f"Error during navigation handling (Attempt {attempt + 1}): {e}"
            last_error_for_retry_prompt = error_msg
            if attempt == MAX_NAV_RETRIES:
                return f"An unexpected error occurred while handling your navigation query: {e}"
                
    return "Failed to handle navigation query after all attempts."
