import os
import sys
import json
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, List

# Try to import error_handler_module with proper fallback
try:
    from .error_handler_module import handle_exception
except ImportError:
    try:
        from error_handler_module import handle_exception
    except ImportError:
        try:
            # Try importing from parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application.error_handler_module import handle_exception
        except ImportError:
            # Define a simple fallback error handler if the module can't be imported
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

# Function to find the data directory dynamically
def find_data_directory() -> Path:
    """
    Find the data directory by starting from the current working directory
    and traversing up if needed.
    
    Returns:
        Path to the data directory
    """
    # Start with the current working directory
    current_dir = Path.cwd()
    
    # Check if data directory exists in current directory
    if (current_dir / "data").exists():
        return current_dir / "data"
    
    # Check if we're in a subdirectory of the project
    parent_dir = current_dir.parent
    if (parent_dir / "data").exists():
        return parent_dir / "data"
    
    # Check if we're in a deeper subdirectory
    grandparent_dir = parent_dir.parent
    if (grandparent_dir / "data").exists():
        return grandparent_dir / "data"
    
    # If we can't find it, create it in the current directory
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

# Try to import pydantic_models with proper fallback
try:
    # First try relative import
    from .pydantic_models import FeedbackReportContentModel, InsightsExtractionModel, \
                                SchemaUnderstandingInsights, QueryConstructionBestPracticesInsights, \
                                CommonErrorsAndCorrectionsInsights, DataValidationTechniquesInsights, \
                                GeneralSQLBestPracticesInsights, FeedbackIteration
except ImportError:
    try:
        # Then try absolute import
        from pydantic_models import FeedbackReportContentModel, InsightsExtractionModel, \
                                  SchemaUnderstandingInsights, QueryConstructionBestPracticesInsights, \
                                  CommonErrorsAndCorrectionsInsights, DataValidationTechniquesInsights, \
                                  GeneralSQLBestPracticesInsights, FeedbackIteration
    except ImportError:
        try:
            # Try importing from parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application.pydantic_models import FeedbackReportContentModel, InsightsExtractionModel, \
                                                 SchemaUnderstandingInsights, QueryConstructionBestPracticesInsights, \
                                                 CommonErrorsAndCorrectionsInsights, DataValidationTechniquesInsights, \
                                                 GeneralSQLBestPracticesInsights, FeedbackIteration
        except ImportError as e:
            print(f"Error importing pydantic_models: {e}", file=sys.stderr)
            # Define minimal fallback classes if needed
            # These are just placeholders to prevent errors
            class FeedbackIteration:
                def __init__(self, user_feedback_text="", corrected_sql_attempt="", corrected_explanation=""):
                    self.user_feedback_text = user_feedback_text
                    self.corrected_sql_attempt = corrected_sql_attempt
                    self.corrected_explanation = corrected_explanation
            
            class FeedbackReportContentModel:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            class SchemaUnderstandingInsights:
                def __init__(self, table_structure_and_relationships=None, field_knowledge=None):
                    self.table_structure_and_relationships = table_structure_and_relationships or []
                    self.field_knowledge = field_knowledge or []
            
            class QueryConstructionBestPracticesInsights:
                def __init__(self, table_joins=None, column_selection=None, filtering_and_sorting=None, numeric_operations=None):
                    self.table_joins = table_joins or []
                    self.column_selection = column_selection or []
                    self.filtering_and_sorting = filtering_and_sorting or []
                    self.numeric_operations = numeric_operations or []
            
            class CommonErrorsAndCorrectionsInsights:
                def __init__(self, schema_misunderstandings=None, logical_errors=None, query_structure_issues=None):
                    self.schema_misunderstandings = schema_misunderstandings or []
                    self.logical_errors = logical_errors or []
                    self.query_structure_issues = query_structure_issues or []
            
            class DataValidationTechniquesInsights:
                def __init__(self, result_verification=None, query_refinement=None):
                    self.result_verification = result_verification or []
                    self.query_refinement = query_refinement or []
            
            class GeneralSQLBestPracticesInsights:
                def __init__(self):
                    pass
            
            class InsightsExtractionModel:
                def __init__(self, title="", introduction="", schema_understanding=None, **kwargs):
                    self.title = title
                    self.introduction = introduction
                    self.schema_understanding = schema_understanding or SchemaUnderstandingInsights()
                    self.query_construction_best_practices = QueryConstructionBestPracticesInsights()
                    self.common_errors_and_corrections = CommonErrorsAndCorrectionsInsights()
                    self.data_validation_techniques = DataValidationTechniquesInsights()
                    self.specific_database_insights_map = {}
                
                def add_insight(self, path, insight, db_name_hint=None):
                    print(f"Warning: Using fallback InsightsExtractionModel.add_insight", file=sys.stderr)

# --- Path Helper Functions ---

def get_memory_base_path() -> Path:
    """Gets the base path for memory files."""
    data_dir = find_data_directory()
    memory_path = data_dir / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)
    return memory_path

def get_approved_queries_path() -> Path:
    """Gets the base path for storing NLQ-SQL JSON pair files."""
    data_dir = find_data_directory()
    approved_queries_dir = data_dir / "Approved_NL2SQL_Pairs"
    approved_queries_dir.mkdir(parents=True, exist_ok=True)
    return approved_queries_dir

def get_vector_store_base_path() -> Path:
    """Gets the base path for LanceDB vector stores."""
    memory_base_path = get_memory_base_path()
    vector_store_path = memory_base_path / "lancedb_stores"
    vector_store_path.mkdir(parents=True, exist_ok=True)
    return vector_store_path

# This function is a temporary helper for postgres_copilot_chat.py's default value
def get_default_memory_base_path_text_for_chat_module() -> str:
    return str(get_memory_base_path())


def ensure_memory_directories():
    """Ensures all necessary memory subdirectories exist based on configuration."""
    mem_base = get_memory_base_path()
    approved_queries_base = get_approved_queries_path()
    vector_store_base = get_vector_store_base_path() # This is memory_base / "lancedb_stores"

    (mem_base / "feedback").mkdir(parents=True, exist_ok=True)
    (mem_base / "insights").mkdir(parents=True, exist_ok=True)
    (mem_base / "schema").mkdir(parents=True, exist_ok=True)
    (mem_base / "conversation_history").mkdir(parents=True, exist_ok=True)
    
    approved_queries_base.mkdir(parents=True, exist_ok=True)
    vector_store_base.mkdir(parents=True, exist_ok=True) # Ensure the parent for lancedb tables exists
    # Individual lancedb table directories within vector_store_base will be handled by vector_store_module

# The ensure_memory_directories() function is now called from the main application
# entry point (postgres_copilot_chat.py) after the configuration is loaded,
# to prevent path issues during initial setup.


def get_insights_filepath(db_name_identifier: str) -> Path:
    """Helper function to construct the filepath for a DB's insights file."""
    return get_memory_base_path() / "insights" / f"{db_name_identifier}_summarized_insights.md"

def get_schema_filepath(db_name_identifier: str) -> str:
    """Constructs the full path for a database's schema JSON file."""
    schema_dir = get_memory_base_path() / "schema"
    return str(schema_dir / f"schema_sampledata_for_{db_name_identifier}.json")

def get_active_tables_filepath(db_name_identifier: str) -> str:
    """Constructs the full path for the active tables file."""
    schema_dir = get_memory_base_path() / "schema"
    return str(schema_dir / f"active_tables_for_{db_name_identifier}.txt")

def get_schema_graph_filepath(db_name_identifier: str) -> Path:
    """Constructs the full path for the schema graph JSON file."""
    schema_dir = get_memory_base_path() / "schema"
    return schema_dir / f"{db_name_identifier}_schema_graph.json"

def get_schema_backup_filepath(db_name_identifier: str) -> Path:
    """Constructs the full path for the schema backup text file."""
    schema_dir = get_memory_base_path() / "schema"
    return schema_dir / f"schema_{db_name_identifier}_backup.txt"

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
    feedback_dir = get_memory_base_path() / "feedback"
    filepath = feedback_dir / filename
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_str)
        return filepath
    except IOError as e:
        handle_exception(e, user_query=f"save_feedback_markdown for {db_name}")
        raise

def read_feedback_file(filepath: str) -> str:
    """Reads the content of a specified feedback markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except IOError as e:
        handle_exception(e, user_query=f"read_feedback_file at {filepath}")
        raise

# --- Cumulative Insights Handling (now per-database) ---

def _format_insights_to_markdown(insights_data: InsightsExtractionModel, db_name_identifier: Optional[str] = None) -> str: # Added db_name_identifier for potential future use in formatting
    """Converts an InsightsExtractionModel object to a markdown string."""
    # Title and intro can be generic or adapted if db_name_identifier is used
    title = insights_data.title
    if db_name_identifier and "{db_name}" in title:
        title = title.replace("{db_name}", ' '.join(word.capitalize() for word in db_name_identifier.replace('_', ' ').split()))
    
    introduction = insights_data.introduction
    if db_name_identifier and "{db_name}" in introduction:
        introduction = introduction.replace("{db_name}", ' '.join(word.capitalize() for word in db_name_identifier.replace('_', ' ').split()))

    md_lines = [title, "", introduction, ""]

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
    
    return "\n".join(md_lines)

def save_or_update_insights(new_insights_data: InsightsExtractionModel, db_name_identifier: str):
    """
    Saves or updates the insights file specific to the given db_name_identifier.
    The merging logic (combining old and new insights) is expected to have happened
    in the insights_module, resulting in `new_insights_data` being the complete model to save.
    """
    ensure_memory_directories()
    insights_filepath = get_insights_filepath(db_name_identifier)

    # The `new_insights_data` IS the complete, merged model from insights_module.
    # The _format_insights_to_markdown can optionally use db_name_identifier if title/intro are templates
    markdown_content = _format_insights_to_markdown(new_insights_data, db_name_identifier)
    try:
        with open(insights_filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    except IOError as e:
        handle_exception(e, user_query=f"save_or_update_insights for {db_name_identifier}")
        raise

def read_insights_file(db_name_identifier: str) -> Optional[str]:
    """Reads the content of the insights file for a specific database."""
    ensure_memory_directories() # Ensure insights dir exists
    insights_filepath = get_insights_filepath(db_name_identifier)
    
    if not os.path.exists(insights_filepath):
        # print(f"Insights file for '{db_name_identifier}' not found at: {insights_filepath}", file=sys.stderr) # Removed
        return None
    try:
        with open(insights_filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except IOError as e:
        handle_exception(e, user_query=f"read_insights_file for {db_name_identifier}")
        return None

# --- Schema and Sample Data Handling ---

def save_schema_data(schema_data: Dict[str, Any], db_name: str) -> str:
    """
    Saves the fetched schema and sample data dictionary as a JSON file.
    Filename will be schema_sampledata_for_{db_name}.json.
    Returns the path to the saved file.
    """
    ensure_memory_directories()
    schema_dir = get_memory_base_path() / "schema"
    filename = f"schema_sampledata_for_{db_name}.json"
    filepath = schema_dir / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2)
        return filepath
    except (IOError, TypeError) as e:
        handle_exception(e, user_query=f"save_schema_data for {db_name}")
        raise

def read_schema_data(db_name: str) -> Optional[Dict[str, Any]]:
    """
    Reads the schema and sample data JSON file for a given db_name.
    Returns the data as a dictionary, or None if the file doesn't exist or an error occurs.
    """
    ensure_memory_directories() # Ensure schema dir exists
    schema_dir = get_memory_base_path() / "schema"
    filename = f"schema_sampledata_for_{db_name}.json"
    filepath = schema_dir / filename
    if not filepath.exists():
        print(f"Schema data file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (IOError, json.JSONDecodeError) as e:
        handle_exception(e, user_query=f"read_schema_data for {db_name}")
        return None

def save_schema_graph(db_name_identifier: str, graph_data: Dict[str, Any]):
    """Saves the generated schema graph to a JSON file."""
    ensure_memory_directories()
    filepath = get_schema_graph_filepath(db_name_identifier)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
    except (IOError, TypeError) as e:
        handle_exception(e, user_query=f"save_schema_graph for {db_name_identifier}")
        raise

def load_schema_graph(db_name_identifier: str) -> Optional[Dict[str, Any]]:
    """Loads the schema graph JSON file for a given database."""
    ensure_memory_directories()
    filepath = get_schema_graph_filepath(db_name_identifier)
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (IOError, json.JSONDecodeError) as e:
        handle_exception(e, user_query=f"load_schema_graph for {db_name_identifier}")
        return None

# Try to import vector_store_module with proper fallback
try:
    # First try relative import
    from . import vector_store_module
except ImportError:
    try:
        # Then try absolute import
        import vector_store_module
    except ImportError:
        try:
            # Try importing from parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from application import vector_store_module
        except ImportError as e:
            print(f"Error importing vector_store_module: {e}", file=sys.stderr)
            # Define a minimal fallback if needed
            class VectorStoreModuleFallback:
                @staticmethod
                def add_nlq_sql_pair(db_name_identifier, nlq, sql):
                    print(f"Warning: Using fallback vector_store_module.add_nlq_sql_pair", file=sys.stderr)
            
            vector_store_module = VectorStoreModuleFallback()

# --- NL2SQL Storage Handling ---

def get_nl2sql_filepath(db_name_identifier: str) -> Path:
    """Helper function to construct the filepath for a DB's NL2SQL JSON file."""
    return get_approved_queries_path() / f"{db_name_identifier}_nl2sql_pairs.json" # Changed filename for clarity

def save_nl2sql_pair(db_name_identifier: str, natural_language_question: str, sql_query: str):
    """
    Saves a natural language question and its corresponding approved SQL query
    to a JSON file named after the database.
    Each entry is a dictionary {"nlq": question, "sql": query}.
    The file will contain a list of these dictionaries.
    """
    ensure_memory_directories() # Ensure NL2SQL_DIR exists
    filepath = get_nl2sql_filepath(db_name_identifier)
    
    new_entry = {"nlq": natural_language_question, "sql": sql_query}
    
    entries: List[Dict[str, str]] = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                entries = json.load(f)
                if not isinstance(entries, list): # Ensure it's a list
                    print(f"Warning: NL2SQL file {filepath} was not a list. Re-initializing.")
                    entries = []
        except (json.JSONDecodeError, IOError) as e:
            handle_exception(e, user_query=f"read_nl2sql_data for {db_name_identifier}")
            entries = []
            
    # Check for duplicates before appending
    is_duplicate = False
    for entry in entries:
        if entry.get("nlq") == natural_language_question and entry.get("sql") == sql_query:
            is_duplicate = True
            break
            
    if not is_duplicate:
        entries.append(new_entry)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            
            # --- Add to Vector Store ---
            try:
                vector_store_module.add_nlq_sql_pair(
                    db_name_identifier=db_name_identifier,
                    nlq=natural_language_question,
                    sql=sql_query
                )
            except Exception as e_vec:
                handle_exception(e_vec, user_query=f"add_nlq_sql_pair to vector store for {db_name_identifier}")
            # --- End Add to Vector Store ---

        except (IOError, TypeError) as e:
            handle_exception(e, user_query=f"save_nl2sql_pair to file for {db_name_identifier}")
            # Optionally re-raise or handle more gracefully
    else:
        print(f"NLQ-SQL pair for '{db_name_identifier}' is a duplicate in JSON, not saving: NLQ='{natural_language_question[:50]}...'", file=sys.stderr)
        # Even if duplicate in JSON, try adding to vector store as it might have its own duplicate check or be a fresh store
        try:
            vector_store_module.add_nlq_sql_pair(
                db_name_identifier=db_name_identifier,
                nlq=natural_language_question,
                sql=sql_query
            )
        except Exception as e_vec_dup:
            handle_exception(e_vec_dup, user_query=f"add_nlq_sql_pair (duplicate) to vector store for {db_name_identifier}")


def read_nl2sql_data(db_name_identifier: str) -> Optional[List[Dict[str, str]]]:
    """
    Reads the NL2SQL JSON file for a given db_name_identifier.
    Returns a list of dictionaries, or None if the file doesn't exist or an error occurs.
    """
    ensure_memory_directories()
    filepath = get_nl2sql_filepath(db_name_identifier)
    if not filepath.exists():
        print(f"NL2SQL data file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            handle_exception(TypeError(f"NL2SQL file content is not a list: {filepath}"), user_query=f"read_nl2sql_data for {db_name_identifier}")
            return None
        return data
    except (IOError, json.JSONDecodeError) as e:
        handle_exception(e, user_query=f"read_nl2sql_data for {db_name_identifier}")
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
    test_db_id = "test_db_insights"
    sample_insights = InsightsExtractionModel(
        title=f"Cumulative Insights for {test_db_id}",
        introduction=f"This is a test insights document for {test_db_id}.",
        schema_understanding=SchemaUnderstandingInsights(
            table_structure_and_relationships=["Users and Orders are related by user_id."]
        )
    )
    sample_insights.add_insight("query_construction_best_practices.table_joins", "Always use explicit JOIN ON conditions.")
    # specific_database_insights_map is within the model, so it's fine.
    sample_insights.add_insight("specific_database_insights_map." + test_db_id, f"The '{test_db_id}' has a special 'audit_log' table.", db_name_hint=test_db_id)
    
    try:
        save_or_update_insights(sample_insights, test_db_id) 
        insights_content = read_insights_file(test_db_id)
        if insights_content:
            insights_file_path_test = get_insights_filepath(test_db_id)
            print(f"Content of insights file ({insights_file_path_test}):\n{insights_content[:300]}...")
        
        # Simulate adding more insights later
        updated_insights = InsightsExtractionModel.model_validate_json(json.loads(sample_insights.model_dump_json())) # Deep copy
        updated_insights.add_insight("common_errors_and_corrections.logical_errors", "Forgetting date ranges is common.")
        updated_insights.add_insight("specific_database_insights_map." + test_db_id, f"The 'audit_log' table in '{test_db_id}' is partitioned daily.", db_name_hint=test_db_id)
        save_or_update_insights(updated_insights, test_db_id)
        insights_content_updated = read_insights_file(test_db_id)
        if insights_content_updated:
            print(f"Updated content of '{test_db_id}' insights file:\n{insights_content_updated[:400]}...")

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

    # Test NL2SQL Storage
    test_db_nl2sql = "test_db_nl2sql"
    try:
        save_nl2sql_pair(test_db_nl2sql, "Show all users", "SELECT * FROM users;")
        save_nl2sql_pair(test_db_nl2sql, "Count active products", "SELECT COUNT(*) FROM products WHERE active = TRUE;")
        save_nl2sql_pair(test_db_nl2sql, "Show all users", "SELECT * FROM users;") # Test duplicate
        
        nl2sql_data = read_nl2sql_data(test_db_nl2sql)
        if nl2sql_data:
            print(f"Retrieved NL2SQL data for {test_db_nl2sql}: {nl2sql_data}")
            assert len(nl2sql_data) == 2 # Check if duplicate was handled
    except Exception as e:
        print(f"Error in NL2SQL storage test: {e}")


    print("memory_module.py testing complete.")
