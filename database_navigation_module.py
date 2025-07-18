import json
import re
import memory_module
from typing import TYPE_CHECKING, Dict, Any, Optional
import litellm
from error_handler_module import handle_exception, SchemaTooLargeError
from pydantic_models import SQLGenerationResponse
from pydantic import BaseModel, Field, ValidationError
import token_utils
if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient

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
            table_names = [name for name, props in raw_schema_and_sample_data.items() if props.get('type') == 'TABLE']
            view_names = [name for name, props in raw_schema_and_sample_data.items() if props.get('type') == 'VIEW']
            
            context_parts = []
            if table_names:
                context_parts.append(f"Tables: {', '.join(table_names)}")
            if view_names:
                context_parts.append(f"Views: {', '.join(view_names)}")

            schema_context_str = (
                f"The connected database ('{current_db_name}') has the following objects ({'; '.join(context_parts)}). "
                f"Full DDL and sample data were loaded at initialization."
            )
        except Exception as e:
            handle_exception(e, user_query, {"detail": "Error processing schema for navigation prompt context"})
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
            schema_tokens = token_utils.count_tokens(schema_context_str, client.model_name, client.provider)
            llm_response_obj = await client._send_message_to_llm(current_call_messages, user_query, schema_tokens)
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
                if not action_decision.sql_query_to_execute or not action_decision.sql_query_to_execute.strip().upper().startswith("SELECT"):
                    return "LLM decided to execute SQL but did not provide a valid SELECT query. Please try rephrasing."
                
                user_feedback_header = (
                    f"{action_decision.explanation_for_user or 'Executing SQL as per your request.'}\n"
                    f"Executing SQL:\n```sql\n{action_decision.sql_query_to_execute}\n```\n"
                )
                
                try:
                    exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": action_decision.sql_query_to_execute})
                    raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj, "execute_postgres_query")
                except Exception as e_tool_call:
                    return f"{user_feedback_header}\n{handle_exception(e_tool_call, user_query, {'tool': 'execute_postgres_query'})}"

                # Handle errors first
                if isinstance(raw_exec_output, str) and "Error:" in raw_exec_output:
                    return f"{user_feedback_header}\nExecution Error: {raw_exec_output}"

                # Handle no results
                is_empty_result = False
                if isinstance(raw_exec_output, str) and "No results found" in raw_exec_output:
                    is_empty_result = True
                elif isinstance(raw_exec_output, list) and not raw_exec_output:
                    is_empty_result = True
                elif isinstance(raw_exec_output, dict) and not raw_exec_output.get("data"):
                    is_empty_result = True

                if is_empty_result:
                    return f"{user_feedback_header}\nExecution Result: Query executed successfully. No results found."

                # If we have results, summarize them
                try:
                    result_json_str = json.dumps(raw_exec_output, indent=2, default=str)
                    
                    summarization_prompt = (
                        f"You are a helpful database assistant. The user asked: \"{user_query}\".\n"
                        f"To answer this, the following SQL query was executed:\n```sql\n{action_decision.sql_query_to_execute}\n```\n"
                        f"The query returned this JSON data, which is a list of items:\n```json\n{result_json_str}\n```\n\n"
                        f"Your task is to summarize **all** of the items in this list in a friendly, natural language paragraph. "
                        f"Do not just summarize the first item. Mention several of the items to give a comprehensive overview. "
                        f"For example, instead of just describing the first product, you could say 'I found several products, including the Corner Desk, a Bolt, and a Cabinet with Doors.'\n"
                        f"Explain what the data represents in the context of the user's question. Be concise and helpful.\n"
                        f"Respond ONLY with the natural language summary. Do not include any other text or markdown."
                    )
                    
                    summary_messages = [{"role": "user", "content": summarization_prompt}]
                    summary_response_obj = await client._send_message_to_llm(summary_messages, user_query)
                    summary_text, _ = await client._process_llm_response(summary_response_obj)
                    
                    return f"{user_feedback_header}\n{summary_text}"

                except Exception as e_summarize:
                    return f"{user_feedback_header}\n{handle_exception(e_summarize, user_query, {'detail': 'Failed to generate natural language summary.'})}\nRaw Result:\n{json.dumps(raw_exec_output, indent=2, default=str)}"


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
                        tool_output = client._extract_mcp_tool_call_output(schema_result_obj, "get_schema_and_sample_data")
                        
                        if isinstance(tool_output, str) and "Error:" in tool_output:
                            return f"{user_feedback_display}\nError fetching schema via tool: {tool_output}"
                        
                        if isinstance(tool_output, dict) and tool_output:
                            current_schema_data_to_use = tool_output
                        # If tool_output is not a dict or empty, current_schema_data_to_use remains None
                        
                    except Exception as tool_e:
                        return f"{user_feedback_display}\n{handle_exception(tool_e, user_query, {'tool': 'get_schema_and_sample_data'})}"

                if current_schema_data_to_use:
                    target_object_name = None
                    # Regex to find a potential object name, possibly qualified with "table" or "view"
                    match = re.search(r"(?:table|view)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", user_query, re.IGNORECASE)
                    if not match:
                        match = re.search(r"([a-zA-Z_][a-zA-Z0-9_.]+)\s+(?:table|view)", user_query, re.IGNORECASE)
                    
                    if match:
                        target_object_name = match.group(1)
                    
                    schema_to_explain_final = current_schema_data_to_use 
                    if target_object_name and target_object_name in current_schema_data_to_use:
                        schema_to_explain_final = {target_object_name: current_schema_data_to_use[target_object_name]}

                    schema_for_llm_str = json.dumps(schema_to_explain_final, indent=2)
                    
                    # Token check before sending to LLM
                    prompt_template_for_explanation = (
                        f"The user originally asked: \"{user_query}\"\n"
                        f"You previously decided that schema information was necessary to answer this. "
                        f"The following schema details (and possibly sample data) were retrieved/accessed for {'object ' + target_object_name if target_object_name and schema_to_explain_final.get(target_object_name) else 'the database'}:\n"
                        f"```json\n__SCHEMA_JSON__\n```\n"
                        f"Your task now is to use this information to provide a concise, natural language answer to the user's original question: \"{user_query}\".\n"
                        f"IMPORTANT: Your response should be **plain text only**. Do NOT output JSON. Do NOT use markdown code blocks like ```json. "
                        f"Extract the relevant information from the schema and explain it clearly. "
                        f"For example, if the user asked about columns in a table, list the column names and perhaps their types. If they asked about what a table contains, describe its purpose based on its columns and sample data (if available).\n"
                        f"Focus on being helpful and directly addressing the query. Avoid technical jargon where possible, or explain it if necessary. "
                        f"Do not mention the process of fetching/accessing data or this internal step. Just provide the answer as a natural language paragraph or bullet points."
                    )
                    
                    available_tokens = token_utils.calculate_available_tokens(prompt_template_for_explanation.replace("__SCHEMA_JSON__", ""), client.model_name, client.provider)
                    schema_tokens = token_utils.count_tokens(schema_for_llm_str, client.model_name, client.provider)

                    if schema_tokens > available_tokens:
                        raise SchemaTooLargeError(client.model_name, schema_tokens, available_tokens)

                    final_prompt_content = prompt_template_for_explanation.replace("__SCHEMA_JSON__", schema_for_llm_str)
                    explanation_prompt_messages = client.conversation_history + [{"role": "user", "content": final_prompt_content}]
                    
                    try:
                        explanation_response_obj = await client._send_message_to_llm(explanation_prompt_messages, user_query, schema_tokens)
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
                        return handle_exception(e_explain, user_query, {'detail': 'Failed to generate schema explanation.'})
                else:
                    return f"{user_feedback_from_llm}\nI attempted to retrieve the schema information, but it appears to be unavailable or not in the expected format. I cannot provide a detailed summary based on this."

            elif action_decision.action_type == 'answer_directly':
                return action_decision.explanation_for_user or "I'm not sure how to answer that directly without more specific database interaction."

            elif action_decision.action_type == 'clarify_question':
                return action_decision.clarification_question_for_user or "Could you please clarify your request?"
            
            else: 
                return f"LLM chose an unknown or invalid action type: {action_decision.action_type}. Please try rephrasing."

        except SchemaTooLargeError as e:
            handle_exception(e, user_query)
            active_tables_filepath = memory_module.get_active_tables_filepath(client.current_db_name_identifier)
            return (
                f"The database schema is too large to fit into the model's context window. "
                f"Please reduce the number of active tables by editing the file at:\n"
                f"'{active_tables_filepath}'\n\n"
                f"Then, try your query again."
            )

        except Exception as e:
            user_message = handle_exception(e, user_query, {"attempt": attempt + 1, "llm_response_text": llm_response_text})
            last_error_for_retry_prompt = f"An error occurred: {user_message}"
            if attempt == MAX_NAV_RETRIES:
                return user_message
                
    return "Failed to handle navigation query after all attempts."
