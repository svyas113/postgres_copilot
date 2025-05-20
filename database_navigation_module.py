import json
from typing import TYPE_CHECKING, Dict, Any, Optional

from pydantic_models import SQLGenerationResponse # Re-using for simple SQL execution if LLM generates it
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from postgres_copilot_chat import GeminiMcpClient # To avoid circular import

# Pydantic model for LLM's decision on how to handle navigation
class NavigationAction(BaseModel):
    action_type: str = Field(description="Type of action to take: 'execute_sql', 'get_schema', 'answer_directly', 'clarify_question'")
    sql_query_to_execute: Optional[str] = Field(None, description="SQL query to execute if action_type is 'execute_sql'.")
    explanation_for_user: Optional[str] = Field(None, description="Explanation or direct answer for the user.")
    clarification_question_for_user: Optional[str] = Field(None, description="A question to ask the user for clarification.")

async def handle_navigation_query(
    client: 'GeminiMcpClient', 
    user_query: str,
    current_db_name: str, # For context, though schema is not directly passed here
    insights_markdown_content: Optional[str],
    raw_schema_and_sample_data: Optional[Dict[str, Any]] # Pass the raw schema for LLM to reference
) -> str:
    """
    Handles natural language queries for database navigation and exploration.
    The LLM decides whether to execute a query, fetch schema, or answer directly.
    """
    if not client.chat or not client.session:
        return "Error: LLM chat or MCP session not available for database navigation."

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
        f"Based on the user's query, decide on an action. Respond ONLY with a single JSON object matching this structure:\n"
        f"```json\n{json.dumps(navigation_action_schema, indent=2)}\n```\n"
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

    MAX_NAV_RETRIES = 1 # Fewer retries for navigation as it's more interactive
    llm_response_text = ""

    for attempt in range(MAX_NAV_RETRIES + 1):
        print(f"Attempting navigation action decision (Attempt {attempt + 1}/{MAX_NAV_RETRIES + 1})...")
        try:
            response = await client.chat.send_message_async(prompt)
            current_llm_response_text = ""
            if response.parts:
                for part_item in response.parts:
                    if hasattr(part_item, 'text') and part_item.text:
                        current_llm_response_text += part_item.text
            llm_response_text = current_llm_response_text.strip()

            if llm_response_text.startswith("```json"):
                llm_response_text = llm_response_text[7:]
            if llm_response_text.endswith("```"):
                llm_response_text = llm_response_text[:-3]
            llm_response_text = llm_response_text.strip()
            
            action_decision = NavigationAction.model_validate_json(llm_response_text)
            print(f"LLM decided action: {action_decision.action_type}")

            # --- Perform the decided action ---
            if action_decision.action_type == 'execute_sql':
                if not action_decision.sql_query_to_execute or not action_decision.sql_query_to_execute.strip().upper().startswith("SELECT"):
                    return "LLM decided to execute SQL but did not provide a valid SELECT query. Please try rephrasing."
                
                user_feedback = f"Understood. Executing SQL: {action_decision.sql_query_to_execute}\n"
                if action_decision.explanation_for_user:
                    user_feedback = f"{action_decision.explanation_for_user}\nExecuting SQL: {action_decision.sql_query_to_execute}\n"
                
                exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": action_decision.sql_query_to_execute})
                raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj)

                if isinstance(raw_exec_output, str) and "Error:" in raw_exec_output:
                    return f"{user_feedback}\nExecution Error: {raw_exec_output}"
                
                result_str = json.dumps(raw_exec_output, indent=2, default=str) if not isinstance(raw_exec_output, str) else raw_exec_output
                return f"{user_feedback}\nExecution Result:\n```json\n{result_str}\n```"

            elif action_decision.action_type == 'get_schema':
                user_feedback = action_decision.explanation_for_user or "Fetching schema and sample data as requested..."
                
                schema_result_obj = await client.session.call_tool("get_schema_and_sample_data", {})
                raw_schema_output = client._extract_mcp_tool_call_output(schema_result_obj)

                if isinstance(raw_schema_output, str) and "Error:" in raw_schema_output:
                    return f"{user_feedback}\nError fetching schema: {raw_schema_output}"
                
                schema_str = json.dumps(raw_schema_output, indent=2, default=str) if not isinstance(raw_schema_output, str) else raw_schema_output
                return f"{user_feedback}\nSchema and Sample Data:\n```json\n{schema_str}\n```"

            elif action_decision.action_type == 'answer_directly':
                return action_decision.explanation_for_user or "I'm not sure how to answer that directly without more specific database interaction."

            elif action_decision.action_type == 'clarify_question':
                return action_decision.clarification_question_for_user or "Could you please clarify your request?"
            
            else:
                return f"LLM chose an unknown action type: {action_decision.action_type}. Please try rephrasing."

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"LLM response validation error for navigation (Attempt {attempt + 1}): {e}")
            if attempt == MAX_NAV_RETRIES:
                return f"Error processing navigation request: Could not understand the desired action. Last LLM response: {llm_response_text}"
            # No specific retry prompt modification here for simplicity, LLM will use original prompt.
        except Exception as e:
            print(f"Error during navigation handling (Attempt {attempt + 1}): {e}")
            if attempt == MAX_NAV_RETRIES:
                return f"An unexpected error occurred while handling your navigation query: {e}"
                
    return "Failed to handle navigation query after all attempts." # Fallback
