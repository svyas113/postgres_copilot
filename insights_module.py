import json
from typing import Optional, Any

# Assuming pydantic_models.py and memory_module.py are accessible
from pydantic_models import InsightsExtractionModel, FeedbackReportContentModel # Feedback for type hint if needed
import memory_module # To call save_or_update_insights and read_insights_file

# This module will interact with the LLM (via postgres_copilot_chat.py's self.chat)
# So, the main function here will likely be called from postgres_copilot_chat.py,
# which has access to the LLM chat session.

async def generate_and_update_insights(
    gemini_mcp_client_instance: Any, # Instance of GeminiMcpClient to access self.chat
    feedback_report_markdown_content: str, 
    current_db_name_hint: str
) -> bool:
    """
    Processes a feedback report to extract insights and updates the
    cumulative summarized_insights.md file.

    Args:
        gemini_mcp_client_instance: The instance of GeminiMcpClient to use its .chat for LLM calls.
        feedback_report_markdown_content: The string content of the new feedback report.
        current_db_name_hint: A hint for the current database name (e.g., "california_schools")
                               to help categorize specific database insights.

    Returns:
        True if insights were successfully generated and saved, False otherwise.
    """
    print("Starting insights generation and update process...")

    # 1. Read existing insights
    existing_insights_md_content = memory_module.read_insights_file()
    if existing_insights_md_content:
        print(f"Read existing insights content ({len(existing_insights_md_content)} chars).")
    else:
        print("No existing insights file found, or it's empty. LLM will generate fresh insights structure.")
        existing_insights_md_content = "" # Pass empty string if no existing insights


    # 2. Prepare prompt for LLM
    # The LLM needs to understand the Pydantic structure of InsightsExtractionModel
    # We can provide the model's JSON schema or a clear description.
    
    # For simplicity in the prompt, we'll describe the task and rely on the LLM's ability
    # to understand the JSON output format based on an example or schema description.
    # Providing the Pydantic model's JSON schema is more robust.
    insights_model_json_schema = InsightsExtractionModel.model_json_schema()

    prompt = f"""
You are an AI assistant tasked with analyzing a SQL query feedback report and updating a cumulative insights document.

**Objective:**
Extract key learnings, schema details, query best practices, common errors, and database-specific insights from the provided "NEW FEEDBACK REPORT CONTENT".
Then, intelligently merge these new insights into the "EXISTING CUMULATIVE INSIGHTS" structure.
The goal is to build a rich, evolving knowledge base.

**Output Format:**
Your final output MUST be a single JSON object that strictly conforms to the following Pydantic model schema for `InsightsExtractionModel`:
```json
{json.dumps(insights_model_json_schema, indent=2)}
```

**Key Instructions for Merging and Extraction:**
- **Uniqueness:** When adding insights (which are typically strings in lists), ensure that identical insights are not duplicated within the same list.
- **Categorization:** Correctly place new insights into the appropriate categories defined by the schema (e.g., `schema_understanding`, `query_construction_best_practices`, `specific_database_insights_map`).
- **`specific_database_insights_map`:**
    - The current database being analyzed is "{current_db_name_hint}".
    - New insights specific to this database should be added to the list associated with the key "{current_db_name_hint}" within `specific_database_insights_map`.
    - If this key doesn't exist in the existing insights, create it.
    - Preserve insights for other databases if they exist in the "EXISTING CUMULATIVE INSIGHTS".
- **Completeness:** Your output JSON should represent the *complete, updated* insights structure, incorporating both old and new information. Do not just output the new insights.
- **Title and Introduction:** Generally, preserve the existing `title` and `introduction` unless the new feedback provides a compelling reason to modify them (unlikely).

**EXISTING CUMULATIVE INSIGHTS (Markdown Format):**
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
    llm_response_json_str = None

    for attempt in range(max_attempts):
        print(f"Attempting LLM call for insights extraction (Attempt {attempt + 1}/{max_attempts})...")
        try:
            # Access the chat object from the passed client instance
            llm_response = await gemini_mcp_client_instance.chat.send_message_async(prompt)
            
            response_text = ""
            if llm_response.parts:
                for part in llm_response.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
            
            # Clean the response: remove markdown code block fences if present
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            llm_response_json_str = cleaned_response_text
            
            # 4. Validate Pydantic Model
            updated_insights_model = InsightsExtractionModel.model_validate_json(llm_response_json_str)
            print("LLM response successfully validated against InsightsExtractionModel.")

            # 5. Save updated insights
            memory_module.save_or_update_insights(updated_insights_model)
            print("Insights successfully updated and saved.")
            return True

        except json.JSONDecodeError as e:
            print(f"Error: LLM response was not valid JSON. Error: {e}")
            print(f"LLM Raw Response Snippet: {llm_response_json_str[:500] if llm_response_json_str else 'N/A'}")
            if attempt == max_attempts - 1:
                print("Max attempts reached for LLM insights generation. Failed.")
                return False
            print("Retrying LLM call for insights...")
        except Exception as e: # Catches Pydantic ValidationError and others
            print(f"Error processing or validating LLM response for insights: {e}")
            print(f"LLM Raw Response Snippet: {llm_response_json_str[:500] if llm_response_json_str else 'N/A'}")
            if attempt == max_attempts - 1:
                print("Max attempts reached for LLM insights generation. Failed.")
                return False
            print("Retrying LLM call for insights...")
            
    return False # Should not be reached if loop logic is correct

if __name__ == '__main__':
    # This module is not meant to be run directly for full functionality
    # as it depends on an active GeminiMcpClient instance for LLM calls.
    # The following is for basic structure testing of memory_module calls if needed.
    print("insights_module.py - This module requires a GeminiMcpClient instance to be fully tested.")
    
    # Example: Test reading insights (doesn't require LLM)
    # ensure_memory_directories() # memory_module does this on import
    # existing_md = memory_module.read_insights_file()
    # if existing_md:
    #     print("\nSuccessfully read existing summarized_insights.md:")
    #     print(existing_md[:500] + "...")
    # else:
    #     print("\nsummarized_insights.md not found or is empty.")
    pass
