import os
import json
import datetime
from typing import Optional, Dict, Any, List

# Assuming pydantic_models.py is in the same directory or accessible in PYTHONPATH
from pydantic_models import FeedbackReportContentModel, InsightsExtractionModel, \
                             SchemaUnderstandingInsights, QueryConstructionBestPracticesInsights, \
                             CommonErrorsAndCorrectionsInsights, DataValidationTechniquesInsights, \
                             GeneralSQLBestPracticesInsights, FeedbackIteration

# --- Directory Setup ---
BASE_MEMORY_DIR = os.path.join(os.getcwd(), "memory")
FEEDBACK_DIR = os.path.join(BASE_MEMORY_DIR, "feedback")
INSIGHTS_DIR = os.path.join(BASE_MEMORY_DIR, "insights")
SCHEMA_DIR = os.path.join(BASE_MEMORY_DIR, "schema")
CONVERSATION_HISTORY_DIR = os.path.join(BASE_MEMORY_DIR, "conversation_history") # Added for completeness

# Path for the cumulative insights file
SUMMARIZED_INSIGHTS_FILE_PATH = os.path.join(INSIGHTS_DIR, "summarized_insights.md")

def ensure_memory_directories():
    """Ensures all necessary memory subdirectories exist."""
    os.makedirs(BASE_MEMORY_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    os.makedirs(INSIGHTS_DIR, exist_ok=True)
    os.makedirs(SCHEMA_DIR, exist_ok=True)
    os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
    print(f"Memory directories ensured: {BASE_MEMORY_DIR} and its subdirectories.")

# Call it once on module load to ensure directories are ready
ensure_memory_directories()


# --- Feedback Report Handling ---

def _format_feedback_report_to_markdown(report_content: FeedbackReportContentModel) -> str:
    """Converts a FeedbackReportContentModel object to a formatted markdown string."""
    md_lines = []
    md_lines.append(f"# Feedback Report for Query: \"{report_content.natural_language_question[:60]}...\"")
    md_lines.append(f"**Timestamp:** {datetime.datetime.now().isoformat()}")
    md_lines.append("\n## 1. Natural Language Question")
    md_lines.append(report_content.natural_language_question)

    md_lines.append("\n## 2. Initial SQL Query Generated")
    md_lines.append("```sql")
    md_lines.append(report_content.initial_sql_query)
    md_lines.append("```")
    md_lines.append("\n### Initial Explanation")
    md_lines.append(report_content.initial_explanation or "N/A")

    if report_content.feedback_iterations:
        md_lines.append("\n## 3. Feedback and Correction Iterations")
        for i, iteration in enumerate(report_content.feedback_iterations):
            md_lines.append(f"\n### Iteration {i+1}")
            md_lines.append(f"**User Feedback:** {iteration.user_feedback_text}")
            md_lines.append("\n**Corrected SQL Attempt:**")
            md_lines.append("```sql")
            md_lines.append(iteration.corrected_sql_attempt)
            md_lines.append("```")
            md_lines.append("\n**Corrected Explanation:**")
            md_lines.append(iteration.corrected_explanation or "N/A")
    else:
        md_lines.append("\n## 3. Feedback and Correction Iterations")
        md_lines.append("No corrective feedback iterations were performed for this query.")


    md_lines.append("\n## 4. Final SQL Query")
    if report_content.final_corrected_sql_query:
        md_lines.append("```sql")
        md_lines.append(report_content.final_corrected_sql_query)
        md_lines.append("```")
        md_lines.append("\n### Final Explanation")
        md_lines.append(report_content.final_explanation or "N/A")
    else: # Should ideally be the initial query if no iterations
        md_lines.append("*(Initial query was considered final)*")
        md_lines.append("```sql")
        md_lines.append(report_content.initial_sql_query) # Repeat initial if no "final"
        md_lines.append("```")
        md_lines.append("\n### Explanation (from initial)")
        md_lines.append(report_content.initial_explanation or "N/A")


    md_lines.append("\n## 5. LLM Analysis of This Query Cycle")
    md_lines.append("\n### Why Initial Query Was Wrong/Suboptimal:")
    md_lines.append(report_content.why_initial_query_was_wrong_or_suboptimal or "N/A or initial query was deemed correct.")
    
    md_lines.append("\n### Why Final Query Works/Is Improved:")
    md_lines.append(report_content.why_final_query_works_or_is_improved or "N/A")

    md_lines.append("\n### Database Insights Learned/Reinforced:")
    if report_content.database_insights_learned_from_this_query:
        for insight in report_content.database_insights_learned_from_this_query:
            md_lines.append(f"- {insight}")
    else:
        md_lines.append("N/A")

    md_lines.append("\n### SQL Lessons Learned/Reinforced:")
    if report_content.sql_lessons_learned_from_this_query:
        for lesson in report_content.sql_lessons_learned_from_this_query:
            md_lines.append(f"- {lesson}")
    else:
        md_lines.append("N/A")
        
    # md_lines.append("\n## 6. Final Results Summary")
    # md_lines.append(report_content.final_results_summary_text or "Execution results not summarized in this report.")

    return "\n".join(md_lines)

def save_feedback_markdown(report_content: FeedbackReportContentModel, db_name: Optional[str] = None) -> str:
    """
    Saves the structured feedback content as a timestamped markdown file.
    Returns the path to the saved file.
    """
    ensure_memory_directories()
    markdown_str = _format_feedback_report_to_markdown(report_content)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Include db_name in filename if provided, for better organization
    filename_prefix = f"feedback_{db_name}_" if db_name else "feedback_"
    filename = f"{filename_prefix}{timestamp}.md"
    filepath = os.path.join(FEEDBACK_DIR, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_str)
        print(f"Feedback report saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving feedback report to {filepath}: {e}")
        raise # Re-raise the exception to be handled by the caller

def read_feedback_file(filepath: str) -> str:
    """Reads the content of a specified feedback markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading feedback file {filepath}: {e}")
        raise

# --- Cumulative Insights Handling (summarized_insights.md) ---

def _format_insights_to_markdown(insights_data: InsightsExtractionModel) -> str:
    """Converts an InsightsExtractionModel object to a markdown string."""
    md_lines = [insights_data.title, "", insights_data.introduction, ""]

    def add_section(title: str, items: List[str]):
        if items: # Only add section if there are items
            md_lines.append(f"## {title}")
            for item in items:
                md_lines.append(f"- {item}")
            md_lines.append("") # Add a blank line after the section

    def add_nested_section(main_title: str, sub_sections_dict: Dict[str, List[str]]):
        # Check if any sub-section has content
        has_content = any(sub_items for sub_items in sub_sections_dict.values())
        if not has_content:
            return

        md_lines.append(f"## {main_title}")
        for sub_title, sub_items in sub_sections_dict.items():
            if sub_items:
                # Convert camelCase or snake_case to Title Case for sub_title
                formatted_sub_title = ' '.join(word.capitalize() for word in sub_title.replace('_', ' ').split())
                md_lines.append(f"### {formatted_sub_title}")
                for item in sub_items:
                    md_lines.append(f"- {item}")
                md_lines.append("")
        md_lines.append("")


    add_nested_section("Schema Understanding", {
        "Table Structure and Relationships": insights_data.schema_understanding.table_structure_and_relationships,
        "Field Knowledge": insights_data.schema_understanding.field_knowledge
    })
    add_nested_section("Query Construction Best Practices", {
        "Table Joins": insights_data.query_construction_best_practices.table_joins,
        "Column Selection": insights_data.query_construction_best_practices.column_selection,
        "Filtering and Sorting": insights_data.query_construction_best_practices.filtering_and_sorting,
        "Numeric Operations": insights_data.query_construction_best_practices.numeric_operations
    })
    add_nested_section("Common Errors and Corrections", {
        "Schema Misunderstandings": insights_data.common_errors_and_corrections.schema_misunderstandings,
        "Logical Errors": insights_data.common_errors_and_corrections.logical_errors,
        "Query Structure Issues": insights_data.common_errors_and_corrections.query_structure_issues
    })
    add_nested_section("Data Validation Techniques", {
        "Result Verification": insights_data.data_validation_techniques.result_verification,
        "Query Refinement": insights_data.data_validation_techniques.query_refinement
    })

    if insights_data.specific_database_insights_map:
        md_lines.append("## Specific Database Insights")
        for db_name_hint, specific_insights_list in insights_data.specific_database_insights_map.items():
            if specific_insights_list:
                # Format db_name_hint (e.g., 'california_schools_db' to 'California Schools DB')
                formatted_db_name = ' '.join(word.capitalize() for word in db_name_hint.replace('_', ' ').split())
                md_lines.append(f"### {formatted_db_name}")
                for item in specific_insights_list:
                    md_lines.append(f"- {item}")
                md_lines.append("")
        md_lines.append("")
        
    add_section("General SQL Best Practices", insights_data.general_sql_best_practices.practices)
    
    return "\n".join(md_lines)

def save_or_update_insights(new_insights_data: InsightsExtractionModel, db_name_for_new_specific_insights: Optional[str] = None):
    """
    Saves or updates the cumulative summarized_insights.md file.
    It merges new insights from new_insights_data into the existing file's structure.
    """
    ensure_memory_directories()
    existing_insights = None
    if os.path.exists(SUMMARIZED_INSIGHTS_FILE_PATH):
        try:
            # This is tricky because we're reading MD and need to parse it back into Pydantic,
            # or merge at the Pydantic object level before writing.
            # For simplicity, the LLM in insights_module will be asked to produce a *complete*
            # InsightsExtractionModel. This function will then just write it.
            # If we want to merge, the LLM in insights_module needs the *old* insights content too.

            # Let's assume insights_module provides a fully formed InsightsExtractionModel
            # that already incorporates old + new.
            pass # For now, this function will just write what it's given.
                 # The merging logic will be in the insights_module (LLM assisted).
        except Exception as e:
            print(f"Warning: Could not parse existing insights file {SUMMARIZED_INSIGHTS_FILE_PATH}. Will overwrite if new data is provided. Error: {e}")
            existing_insights = InsightsExtractionModel() # Start fresh
    else:
        existing_insights = InsightsExtractionModel() # Create a new one if file doesn't exist

    # The `new_insights_data` IS the complete, merged model from insights_module.
    final_insights_model_to_save = new_insights_data

    markdown_content = _format_insights_to_markdown(final_insights_model_to_save)
    try:
        with open(SUMMARIZED_INSIGHTS_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Insights saved/updated at: {SUMMARIZED_INSIGHTS_FILE_PATH}")
    except Exception as e:
        print(f"Error saving insights to {SUMMARIZED_INSIGHTS_FILE_PATH}: {e}")
        raise

def read_insights_file() -> Optional[str]:
    """Reads the content of the summarized_insights.md file."""
    ensure_memory_directories() # Ensure insights dir exists
    if not os.path.exists(SUMMARIZED_INSIGHTS_FILE_PATH):
        return None
    try:
        with open(SUMMARIZED_INSIGHTS_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading insights file {SUMMARIZED_INSIGHTS_FILE_PATH}: {e}")
        return None # Return None on error to indicate it's not available

# --- Schema and Sample Data Handling ---

def save_schema_data(schema_data: Dict[str, Any], db_name: str) -> str:
    """
    Saves the fetched schema and sample data dictionary as a JSON file.
    Filename will be schema_sampledata_for_{db_name}.json.
    Returns the path to the saved file.
    """
    ensure_memory_directories()
    filename = f"schema_sampledata_for_{db_name}.json"
    filepath = os.path.join(SCHEMA_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2)
        print(f"Schema and sample data saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving schema data to {filepath}: {e}")
        raise

def read_schema_data(db_name: str) -> Optional[Dict[str, Any]]:
    """
    Reads the schema and sample data JSON file for a given db_name.
    Returns the data as a dictionary, or None if the file doesn't exist or an error occurs.
    """
    ensure_memory_directories() # Ensure schema dir exists
    filename = f"schema_sampledata_for_{db_name}.json"
    filepath = os.path.join(SCHEMA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Schema data file not found: {filepath}")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading schema data from {filepath}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    print("Testing memory_module.py...")
    ensure_memory_directories()

    # Test Feedback Report
    sample_iteration = FeedbackIteration(
        user_feedback_text="The query missed the filter for active users.",
        corrected_sql_attempt="SELECT * FROM users WHERE status = 'active';",
        corrected_explanation="Added a WHERE clause to filter for active users."
    )
    sample_report = FeedbackReportContentModel(
        natural_language_question="Get all active users",
        initial_sql_query="SELECT * FROM users;",
        initial_explanation="Retrieves all users.",
        feedback_iterations=[sample_iteration],
        final_corrected_sql_query="SELECT * FROM users WHERE status = 'active';",
        final_explanation="Added a WHERE clause to filter for active users.",
        why_initial_query_was_wrong_or_suboptimal="Initial query did not filter by status.",
        why_final_query_works_or_is_improved="Final query correctly filters for active users.",
        database_insights_learned_from_this_query=["The 'users' table has a 'status' column."],
        sql_lessons_learned_from_this_query=["Always check for filtering conditions in the NLQ."]
    )
    try:
        feedback_file = save_feedback_markdown(sample_report, db_name="test_db")
        print(f"Test feedback file created: {feedback_file}")
        content = read_feedback_file(feedback_file)
        # print(f"Content of {feedback_file}:\n{content[:300]}...") # Print snippet
    except Exception as e:
        print(f"Error in feedback test: {e}")

    # Test Insights
    sample_insights = InsightsExtractionModel(
        introduction="This is a test insights document for test_db.",
        schema_understanding=SchemaUnderstandingInsights(
            table_structure_and_relationships=["Users and Orders are related by user_id."]
        )
    )
    sample_insights.add_insight("query_construction_best_practices.table_joins", "Always use explicit JOIN ON conditions.")
    sample_insights.add_insight("specific_database_insights_map.test_db", "The 'test_db' has a special 'audit_log' table.", db_name_hint="test_db")
    
    try:
        save_or_update_insights(sample_insights) # This will create/overwrite
        insights_content = read_insights_file()
        if insights_content:
            print(f"Content of insights file ({SUMMARIZED_INSIGHTS_FILE_PATH}):\n{insights_content[:300]}...")
        
        # Simulate adding more insights later (would typically come from LLM processing a new feedback)
        updated_insights = InsightsExtractionModel.model_validate_json(json.loads(sample_insights.model_dump_json())) # Deep copy
        updated_insights.add_insight("common_errors_and_corrections.logical_errors", "Forgetting date ranges is common.")
        updated_insights.add_insight("specific_database_insights_map.test_db", "The 'audit_log' table is partitioned daily.", db_name_hint="test_db")
        save_or_update_insights(updated_insights)
        insights_content_updated = read_insights_file()
        if insights_content_updated:
            print(f"Updated content of insights file:\n{insights_content_updated[:400]}...")

    except Exception as e:
        print(f"Error in insights test: {e}")

    # Test Schema Data
    sample_schema = {
        "users": {
            "schema": "CREATE TABLE users (id INT, name VARCHAR(100));",
            "sample_data": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        }
    }
    try:
        schema_file = save_schema_data(sample_schema, "test_db_schema")
        print(f"Test schema file created: {schema_file}")
        retrieved_schema = read_schema_data("test_db_schema")
        if retrieved_schema:
            print(f"Retrieved schema for test_db_schema: {retrieved_schema}")
    except Exception as e:
        print(f"Error in schema data test: {e}")

    print("memory_module.py testing complete.")
