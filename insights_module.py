import json
from typing import Optional, Any
from pydantic import ValidationError

# Assuming pydantic_models.py and memory_module.py are accessible
from pydantic_models import InsightsExtractionModel, FeedbackReportContentModel # Feedback for type hint if needed
import memory_module # To call save_or_update_insights and read_insights_file

# This module will interact with the LLM (via postgres_copilot_chat.py's _send_message_to_llm method)
# So, the main function here will be called from postgres_copilot_chat.py,
# which has access to the LiteLLM client instance.

async def generate_and_update_insights(
    litellm_mcp_client_instance: Any, # Instance of LiteLLMMcpClient
    feedback_report_markdown_content: str,
    db_name_identifier: str
) -> bool:
    """
    Processes a feedback report to extract insights and updates the
    insights file specific to the given db_name_identifier using LiteLLM.

    Args:
        litellm_mcp_client_instance: The instance of LiteLLMMcpClient.
        feedback_report_markdown_content: The string content of the new feedback report.
        db_name_identifier: The identifier for the current database.

    Returns:
        True if insights were successfully generated and saved, False otherwise.
    """
    # print(f"Starting insights generation and update process for database: {db_name_identifier}...") # User sees main wait message

    # 1. Read existing insights for the specific database
    existing_insights_md_content = memory_module.read_insights_file(db_name_identifier)
    # if existing_insights_md_content: # Debug
    #     print(f"Read existing insights for '{db_name_identifier}' ({len(existing_insights_md_content)} chars).")
    # else: # Debug
    #     print(f"No existing insights file found for '{db_name_identifier}', or it's empty. LLM will generate fresh insights structure for this DB.")
    if not existing_insights_md_content:
        existing_insights_md_content = "" 


    # 2. Prepare prompt for LLM
    # The LLM needs to understand the Pydantic structure of InsightsExtractionModel
    # We can provide the model's JSON schema or a clear description.
    
    # For simplicity in the prompt, we'll describe the task and rely on the LLM's ability
    # to understand the JSON output format based on an example or schema description.
    # Providing the Pydantic model's JSON schema is more robust.
    insights_model_json_schema = InsightsExtractionModel.model_json_schema()

    prompt = f"""
You are an AI assistant tasked with analyzing a SQL query feedback report and updating an insights document for a specific database.

**Objective:**
Extract key learnings, schema details, query best practices, common errors, and database-specific insights from the provided "NEW FEEDBACK REPORT CONTENT" for the database '{db_name_identifier}'.
Then, intelligently merge these new insights into the "EXISTING INSIGHTS FOR '{db_name_identifier}'" structure.
The goal is to build a rich, evolving knowledge base for this specific database.

**Output Format:**
Your final output MUST be a single JSON object that strictly conforms to the following Pydantic model schema for `InsightsExtractionModel`:
```json
{json.dumps(insights_model_json_schema, indent=2)}
```

**Key Instructions for Merging and Extraction:**
- **Uniqueness:** When adding insights (which are typically strings in lists), ensure that identical insights are not duplicated within the same list.
- **Categorization:** Correctly place new insights into the appropriate categories defined by the schema (e.g., `schema_understanding`, `query_construction_best_practices`, `specific_database_insights_map`).
- **`specific_database_insights_map`:**
    - The current database being analyzed is "{db_name_identifier}".
    - New insights specific to this database should be added to the list associated with the key "{db_name_identifier}" within `specific_database_insights_map`.
    - If this key doesn't exist in the existing insights for this DB, create it.
    - This map within the model for *this specific database's insights file* should primarily, if not exclusively, contain insights for `{db_name_identifier}`. If merging from a very old global file, be mindful.
- **Completeness:** Your output JSON should represent the *complete, updated* insights structure for '{db_name_identifier}', incorporating both old (from this DB's file) and new information. Do not just output the new insights.
- **Title and Introduction:** The `title` and `introduction` in the output JSON should be appropriate for an insights document specifically for the database '{db_name_identifier}'. You can adapt the existing ones or generate new ones if the existing ones are too generic or from a previous global file. For example, title could be "Insights for {db_name_identifier}".

**EXISTING INSIGHTS FOR '{db_name_identifier}' (Markdown Format, may be empty if new DB):**
```markdown
{existing_insights_md_content}
```

**NEW FEEDBACK REPORT CONTENT (Markdown Format):**
```markdown
{feedback_report_markdown_content}
```

Now, based on both the existing insights and the new feedback report, generate the complete, updated JSON object conforming to the `InsightsExtractionModel` schema.
Your entire response should be ONLY this JSON object.
"""

    # 3. Call LLM (via the client instance)
    # This is a simplified loop; a more robust one would handle retries for validation.
    max_attempts = 2
    llm_response_json_str = None # To store the string from LLM
    last_error_for_retry_prompt = ""

    # Start with a copy of the client's conversation history
    messages_for_llm = litellm_mcp_client_instance.conversation_history[:]

    for attempt in range(max_attempts):
        # print(f"Attempting LLM call for insights extraction (Attempt {attempt + 1}/{max_attempts})...") # Internal Detail
        
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to generate the insights JSON again, adhering to the schema.\n"
                f"Original request details remain the same (existing insights, new feedback report, and db_name_identifier: {db_name_identifier}).\n"
                f"Respond ONLY with the complete, updated JSON object for `InsightsExtractionModel`."
            )
        
        # Use a temporary list for the call to avoid polluting main history with retries until success
        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]

        try:
            # Call LLM using the client's method. We expect JSON, so no tools.
            llm_response_obj = await litellm_mcp_client_instance._send_message_to_llm(current_call_messages)
            # _send_message_to_llm adds the user prompt to client.conversation_history
            # _process_llm_response will add the assistant's response and return the text content
            
            llm_response_json_str, tool_calls_made = await litellm_mcp_client_instance._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON response for insights directly."
                if attempt == max_attempts - 1:
                    # print("Max attempts reached. LLM attempted tool call instead of providing JSON for insights.") # Keep for debug
                    return False
                # print("Retrying LLM call for insights...") # Debug
                continue
            
            cleaned_response_text = llm_response_json_str.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if not cleaned_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON output for insights."
                if attempt == max_attempts - 1:
                    # print("Max attempts reached. LLM provided an empty response for insights.") # Debug
                    return False
                # print("Retrying LLM call for insights...") # Debug
                continue
            
            llm_response_json_str = cleaned_response_text 
            
            updated_insights_model = InsightsExtractionModel.model_validate_json(llm_response_json_str)
            # print("LLM response successfully validated against InsightsExtractionModel.") # Internal Detail

            memory_module.save_or_update_insights(updated_insights_model, db_name_identifier)
            # print(f"Insights for '{db_name_identifier}' successfully updated and saved.") # User sees main message
            return True

        except (json.JSONDecodeError, ValidationError) as e: 
            error_message = f"Error: LLM response for insights was not valid JSON or failed Pydantic validation. Error: {e}"
            # print(error_message) # Keep for debugging
            # print(f"LLM Raw Response Snippet: {llm_response_json_str[:500] if llm_response_json_str else 'N/A'}") # Debug
            last_error_for_retry_prompt = f"{error_message}\nLLM Raw Response Snippet: {llm_response_json_str[:200] if llm_response_json_str else 'N/A'}"
            if attempt == max_attempts - 1:
                # print("Max attempts reached for LLM insights generation. Failed.") # Debug
                return False
            # print("Retrying LLM call for insights...") # Debug
        except Exception as e: 
            error_message = f"An unexpected error occurred processing LLM response for insights: {e}"
            # print(error_message) # Keep for debugging
            # print(f"LLM Raw Response Snippet: {llm_response_json_str[:500] if llm_response_json_str else 'N/A'}") # Debug
            last_error_for_retry_prompt = error_message
            if attempt == max_attempts - 1:
                # print("Max attempts reached for LLM insights generation due to unexpected error. Failed.") # Debug
                return False
            # print("Retrying LLM call for insights...") # Debug
            
    return False

if __name__ == '__main__':
    # This module is not meant to be run directly for full functionality
    # as it depends on an active LiteLLMMcpClient instance for LLM calls.
    print("insights_module.py - This module requires a LiteLLMMcpClient instance to be fully tested.")
    
    # Example: Test reading insights (doesn't require LLM)
    # memory_module.ensure_memory_directories() # memory_module does this on import
    # existing_md = memory_module.read_insights_file()
    # if existing_md:
    #     print("\nSuccessfully read existing summarized_insights.md:")
    #     print(existing_md[:500] + "...")
    # else:
    #     print("\nsummarized_insights.md not found or is empty.")
    pass
