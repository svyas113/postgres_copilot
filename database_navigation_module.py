import json
import re # Added for extracting table name
from typing import TYPE_CHECKING, Dict, Any, Optional

from pydantic_models import SQLGenerationResponse # Re-using for simple SQL execution if LLM generates it
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient # To avoid circular import

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
            # Provide a snippet or summary to keep the prompt manageable,
            # or let LLM decide to use get_schema_and_sample_data tool.
            # For now, let's provide a hint that it exists.
            table_names = list(raw_schema_and_sample_data.keys())
            schema_context_str = (
                f"The connected database ('{current_db_name}') has the following tables (details can be fetched using tools): {', '.join(table_names)}. "
                f"Full DDL and sample data were loaded at initialization."
            )
        except Exception as e:
            print(f"Error processing schema for navigation prompt context: {e}")
            # Fallback if schema_and_sample_data is not as expected
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
    llm_response_text = "" # To store the final text part of LLM response
    last_error_for_retry_prompt = ""

    # Start with a copy of the client's conversation history
    messages_for_llm = client.conversation_history[:]

    for attempt in range(MAX_NAV_RETRIES + 1):
        # print(f"Attempting navigation action decision (Attempt {attempt + 1}/{MAX_NAV_RETRIES + 1})...") # Internal Detail
        
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to decide the navigation action again, adhering to the JSON schema.\n"
                f"Original user query: \"{user_query}\". Database: '{current_db_name}'.\n"
                f"Respond ONLY with the `NavigationAction` JSON object."
            )

        # Use a temporary list for the call
        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]
        
        try:
            # We expect JSON, so no tools passed to _send_message_to_llm for this specific decision-making call.
            # The LLM is deciding *whether* to use tools, not using them yet.
            llm_response_obj = await client._send_message_to_llm(current_call_messages)
            # _send_message_to_llm adds user prompt to history.
            # _process_llm_response adds assistant response and returns text.
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj)

            if tool_calls_made:
                # This is unexpected, as we asked for a JSON decision.
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON decision directly."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM attempted tool call instead of providing JSON decision for navigation."
                # print("Retrying navigation decision...") # Debug
                continue

            if llm_response_text.startswith("```json"):
                llm_response_text = llm_response_text[7:]
            if llm_response_text.endswith("```"):
                llm_response_text = llm_response_text[:-3]
            llm_response_text = llm_response_text.strip()

            if not llm_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON decision."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM provided an empty response for navigation decision."
                # print("Retrying navigation decision...") # Debug
                continue
            
            action_decision = NavigationAction.model_validate_json(llm_response_text)
            # print(f"LLM decided action: {action_decision.action_type}") # Internal Detail

            # --- Perform the decided action ---
            if action_decision.action_type == 'execute_sql':
                if not action_decision.sql_query_to_execute or not action_decision.sql_query_to_execute.strip().upper().startswith("SELECT"):
                    return "LLM decided to execute SQL but did not provide a valid SELECT query. Please try rephrasing."
                
                # Improved user feedback to include the SQL query
                user_feedback = (
                    f"{action_decision.explanation_for_user or 'Executing SQL as per your request.'}\n"
                    f"Executing SQL:\n```sql\n{action_decision.sql_query_to_execute}\n```\n"
                )
                
                try:
                    exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": action_decision.sql_query_to_execute})
                    raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj)
                except Exception as e_tool_call:
                    return f"{user_feedback}\nError during tool execution: {e_tool_call}"

                # Process the raw_exec_output for better user presentation
                if isinstance(raw_exec_output, str):
                    if "Error:" in raw_exec_output:
                        return f"{user_feedback}\nExecution Error: {raw_exec_output}"
                    elif raw_exec_output == "meta=None content=[] isError=False":
                        # Specific handling for the "no results" string from the tool
                        # Check if the query was about views/functions to give a more specific message
                        query_lower = action_decision.sql_query_to_execute.lower()
                        if "information_schema.views" in query_lower or "information_schema.routines" in query_lower:
                            return f"{user_feedback}\nExecution Result: Query executed successfully. No user-defined views or functions found."
                        return f"{user_feedback}\nExecution Result: Query executed successfully. No results found."
                    else: # Other string output
                        return f"{user_feedback}\nExecution Result:\n{raw_exec_output}"
                elif isinstance(raw_exec_output, dict):
                    if raw_exec_output.get("error"):
                        return f"{user_feedback}\nExecution Error: {raw_exec_output['error']}"
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
                    else: # Unrecognized dictionary structure
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
                else: # Other unexpected types
                    return f"{user_feedback}\nExecution Result (unknown data type: {type(raw_exec_output)}):\n{str(raw_exec_output)}"

            elif action_decision.action_type == 'get_schema':
                user_feedback = action_decision.explanation_for_user or "Fetching schema and sample data as requested..."
                
                # Here, the LLM has decided to call 'get_schema_and_sample_data'.
                # We need to simulate this or directly call the MCP tool.
                # For LiteLLM, if the LLM itself was supposed to make the tool call,
                # the previous _process_llm_response would have handled it.
                # Since we are manually interpreting the JSON, we call the tool directly.
                try:
                    schema_result_obj = await client.session.call_tool("get_schema_and_sample_data", {})
                    raw_schema_output = client._extract_mcp_tool_call_output(schema_result_obj)

                    # Add this tool interaction to history manually if needed, or let the next LLM call summarize.
                    # For now, we just return the result.
                    # client.conversation_history.append({"role": "tool", "name": "get_schema_and_sample_data", "content": str(raw_schema_output)}) # LiteLLM client handles history

                    if isinstance(raw_schema_output, str) and "Error:" in raw_schema_output:
                        return f"{user_feedback}\nError fetching schema: {raw_schema_output}"

                    # --- Second LLM call to interpret the schema and answer the original question ---
                    if isinstance(raw_schema_output, dict) and raw_schema_output: # Ensure we have a non-empty dict
                        # Try to find if a specific table was asked about in the original user_query
                        target_table_name = None
                        # Simple extraction: look for "table <name>" or "<name> table"
                        match_table_word = re.search(r"table\s+([a-zA-Z_][a-zA-Z0-9_]*)", user_query, re.IGNORECASE)
                        if match_table_word:
                            target_table_name = match_table_word.group(1)
                        else:
                            match_word_table = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+table", user_query, re.IGNORECASE)
                            if match_word_table:
                                target_table_name = match_word_table.group(1)
                        
                        schema_to_explain = raw_schema_output # Default to full schema
                        if target_table_name and target_table_name in raw_schema_output:
                            schema_to_explain = {target_table_name: raw_schema_output[target_table_name]}
                            # print(f"Extracted target table '{target_table_name}' for schema explanation.") # Debug
                        else:
                            # print(f"No specific table found in query '{user_query}' or not in schema, LLM will use full schema context.") # Debug
                            pass

                        schema_for_llm_str = json.dumps(schema_to_explain, indent=2)
                        
                        explanation_prompt_messages = client.conversation_history + [
                            {"role": "user", 
                             "content": f"The user originally asked: \"{user_query}\"\n"
                                        f"You decided to fetch schema information. Here is the relevant schema and sample data that was retrieved:\n"
                                        f"```json\n{schema_for_llm_str}\n```\n"
                                        f"Based on this retrieved information, please now directly answer the user's original question: \"{user_query}\"\n"
                                        f"Provide a concise and natural language explanation. Do not refer to the process of fetching schema, just answer the question."
                            }
                        ]
                        
                        try:
                            # print("Making second LLM call to explain fetched schema...") # User sees wait message from main loop
                            explanation_response_obj = await client._send_message_to_llm(explanation_prompt_messages)
                            final_answer, _ = await client._process_llm_response(explanation_response_obj)
                            return f"{user_feedback}\n{final_answer}" 
                        except Exception as e_explain:
                            # print(f"Error during second LLM call for schema explanation: {e_explain}") # Keep for debugging
                            schema_str = json.dumps(raw_schema_output, indent=2, default=str)
                            return f"{user_feedback}\nI fetched the schema, but had trouble explaining it. Here's the raw data:\n```json\n{schema_str}\n```"
                    else: 
                        schema_str = str(raw_schema_output) 
                        return f"{user_feedback}\nSchema and Sample Data:\n```\n{schema_str}\n```"

                except Exception as tool_e:
                    return f"{user_feedback}\nError calling get_schema_and_sample_data tool: {tool_e}"

            elif action_decision.action_type == 'answer_directly':
                return action_decision.explanation_for_user or "I'm not sure how to answer that directly without more specific database interaction."

            elif action_decision.action_type == 'clarify_question':
                return action_decision.clarification_question_for_user or "Could you please clarify your request?"
            
            else: # Should not happen if action_type is validated by Pydantic
                return f"LLM chose an unknown or invalid action type: {action_decision.action_type}. Please try rephrasing."

        except (ValidationError, json.JSONDecodeError) as e:
            error_msg = f"LLM response validation error for navigation (Attempt {attempt + 1}): {e}"
            # print(error_msg) # Keep for debugging
            last_error_for_retry_prompt = f"{error_msg}\nProblematic response: {llm_response_text[:200]}"
            if attempt == MAX_NAV_RETRIES:
                return f"Error processing navigation request: Could not understand the desired action. Last LLM response: {llm_response_text}"
        except Exception as e:
            error_msg = f"Error during navigation handling (Attempt {attempt + 1}): {e}"
            # print(error_msg) # Keep for debugging
            last_error_for_retry_prompt = error_msg
            if attempt == MAX_NAV_RETRIES:
                return f"An unexpected error occurred while handling your navigation query: {e}"
                
    return "Failed to handle navigation query after all attempts."
