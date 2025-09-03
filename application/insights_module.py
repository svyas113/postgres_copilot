import json
from typing import Optional, Any
from pydantic import ValidationError

# Try to import error_handler_module with proper fallback
try:
    from .error_handler_module import handle_exception, display_message
except ImportError:
    try:
        from error_handler_module import handle_exception, display_message
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application.error_handler_module import handle_exception, display_message
        except ImportError:
            # Define simple fallback error handlers if the module can't be imported
            def handle_exception(e, user_query=None, context=None):
                """Simple error handling function"""
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
                
            def display_message(message, level="INFO", log=True):
                """Simple message display function"""
                print(f"[{level}] {message}")

# Try to import pydantic_models with proper fallback
try:
    from .pydantic_models import InsightsExtractionModel, FeedbackReportContentModel
except ImportError:
    try:
        from pydantic_models import InsightsExtractionModel, FeedbackReportContentModel
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application.pydantic_models import InsightsExtractionModel, FeedbackReportContentModel
        except ImportError as e:
            print(f"Error importing pydantic_models: {e}", file=sys.stderr)
            # Define minimal fallback classes if needed
            # These are just placeholders to prevent errors
            class FeedbackReportContentModel:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            class InsightsExtractionModel:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                
                @classmethod
                def model_json_schema(cls):
                    return {}
                
                @classmethod
                def model_validate_json(cls, json_str):
                    return cls()

# Import other modules with proper fallback
try:
    from . import memory_module
except ImportError:
    try:
        import memory_module
    except ImportError:
        try:
            # Try importing from parent directory
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application import memory_module
        except ImportError as e:
            print(f"Error importing memory_module: {e}", file=sys.stderr)
            # Define a minimal fallback if needed
            class MemoryModuleFallback:
                @staticmethod
                def read_insights_file(*args, **kwargs):
                    return ""
                
                @staticmethod
                def save_or_update_insights(*args, **kwargs):
                    pass
                
                @staticmethod
                def load_schema_graph(*args, **kwargs):
                    return {}
            
            memory_module = MemoryModuleFallback()

# Define a simple token counting function if token_utils is not available
def count_tokens(text, model_name=None, provider=None):
    """Simple token counting function that estimates tokens based on words"""
    if not text:
        return 0
    # A very rough approximation: 1 token ≈ 0.75 words
    words = len(text.split())
    return int(words * 0.75)

async def generate_and_update_insights(
    litellm_mcp_client_instance: Any, # Instance of LiteLLMMcpClient
    feedback_report_markdown_content: str,
    db_name_identifier: str,
    sql_query: str = None,  # Add SQL query parameter
    user_feedback: str = None  # Add user feedback parameter
) -> bool:
    """
    Processes a feedback report to extract insights and updates the
    insights file specific to the given db_name_identifier using LiteLLM.

    Args:
        litellm_mcp_client_instance: The instance of LiteLLMMcpClient.
        feedback_report_markdown_content: The string content of the new feedback report.
        db_name_identifier: The identifier for the current database.
        sql_query: Optional SQL query for context.
        user_feedback: Optional user feedback for context.

    Returns:
        True if insights were successfully generated and saved, False otherwise.
    """
    # 1. Read existing insights for the specific database
    existing_insights_md_content = memory_module.read_insights_file(db_name_identifier)
    if not existing_insights_md_content:
        existing_insights_md_content = ""

    # 2. Prepare prompt for LLM
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

    for attempt in range(max_attempts):
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to generate the insights JSON again, adhering to the schema.\n"
                f"Original request details remain the same (existing insights, new feedback report, and db_name_identifier: {db_name_identifier}).\n"
                f"Respond ONLY with the complete, updated JSON object for `InsightsExtractionModel`."
            )
        
        # Create a message for the LLM
        current_call_messages = [{"role": "user", "content": current_user_prompt_content}]

        try:
            # Call LLM using the client's chat method directly
            # We expect JSON, so we'll specify response_format for OpenAI models
            response_format = None
            if litellm_mcp_client_instance.model_name.startswith("gpt-") or litellm_mcp_client_instance.model_name.startswith("openai/"):
                response_format = {"type": "json_object"}
            
            llm_response_obj = await litellm_mcp_client_instance.chat(
                messages=current_call_messages,
                response_format=response_format
            )
            
            # Extract the response text from the LLM response object
            if llm_response_obj and llm_response_obj.choices and len(llm_response_obj.choices) > 0:
                llm_response_json_str = llm_response_obj.choices[0].message.content
            else:
                last_error_for_retry_prompt = "Received an empty response from the LLM. Please try again."
                if attempt == max_attempts - 1:
                    return False
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
                    return False
                continue
            
            llm_response_json_str = cleaned_response_text 
            
            updated_insights_model = InsightsExtractionModel.model_validate_json(llm_response_json_str)

            memory_module.save_or_update_insights(updated_insights_model, db_name_identifier)
            return True

        except Exception as e: 
            user_message = handle_exception(e, user_query=f"Insights generation for {db_name_identifier}", context={"attempt": attempt + 1, "llm_response_text": llm_response_json_str})
            last_error_for_retry_prompt = f"An error occurred: {user_message}"
            if attempt == max_attempts - 1:
                return False
            
    return False

if __name__ == '__main__':
    # This module is not meant to be run directly for full functionality
    # as it depends on an active LiteLLMMcpClient instance for LLM calls.
    display_message("insights_module.py - This module requires a LiteLLMMcpClient instance to be fully tested.", level="WARNING")
