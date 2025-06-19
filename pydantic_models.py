from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any, Dict

class SQLGenerationResponse(BaseModel):
    """
    Represents the structured response expected from the LLM for SQL generation.
    """
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    error_message: Optional[str] = None # If LLM itself detects an issue or cannot generate

    @validator('sql_query', pre=True, always=True)
    def ensure_sql_query_is_select(cls, value):
        if value and not value.strip().upper().startswith("SELECT"):
            # This validation might be too strict if we ever want other types of queries.
            # For now, as per user instructions, it must be SELECT.
            # Consider if this should be an error or a modification.
            # For now, let's raise a ValueError that the Pydantic loop can catch.
            # raise ValueError("SQL query must start with SELECT.")
            # Or, let the main logic handle this if it's a soft requirement.
            # For Pydantic, it's better to validate the received data.
            # If the LLM is *instructed* to only return SELECT, this validates its adherence.
            pass # Let's assume the LLM will follow instructions, or handle non-SELECT upstream.
        return value

class FeedbackIteration(BaseModel):
    """
    Represents a single iteration of feedback provided by the user
    and the LLM's subsequent correction.
    """
    user_feedback_text: str
    corrected_sql_attempt: str
    corrected_explanation: str
    # Optional: Add fields for execution results of this specific attempt if needed later

class FeedbackReportContentModel(BaseModel):
    """
    Structures the entire content of a feedback report for a single natural language query.
    This model will be filled by the LLM.
    """
    natural_language_question: str
    initial_sql_query: str
    initial_explanation: str
    
    feedback_iterations: List[FeedbackIteration] = Field(default_factory=list, description="Chronological list of feedback and correction attempts.")
    
    final_corrected_sql_query: Optional[str] = Field(None, description="The very last SQL query after all feedback iterations.")
    final_explanation: Optional[str] = Field(None, description="The explanation for the final_corrected_sql_query.")

    # LLM-generated analysis sections based on the whole process for THIS query
    why_initial_query_was_wrong_or_suboptimal: Optional[str] = Field(None, description="LLM's analysis of what was wrong with the initial query, if applicable.")
    why_final_query_works_or_is_improved: Optional[str] = Field(None, description="LLM's analysis of why the final query is correct or improved.")
    database_insights_learned_from_this_query: List[str] = Field(default_factory=list, description="Specific database insights learned or reinforced from this query-feedback cycle.")
    sql_lessons_learned_from_this_query: List[str] = Field(default_factory=list, description="General SQL lessons learned or reinforced from this cycle.")
    # final_results_summary_text: Optional[str] = Field(None, description="A brief text summary of the outcome of executing the final_corrected_sql_query.")


# Models for the cumulative summarized_insights.md file

class SchemaUnderstandingInsights(BaseModel):
    table_structure_and_relationships: List[str] = Field(default_factory=list, description="Observations about table structures and how they relate.")
    field_knowledge: List[str] = Field(default_factory=list, description="Specific knowledge about fields, their meanings, and authoritative sources.")

class QueryConstructionBestPracticesInsights(BaseModel):
    table_joins: List[str] = Field(default_factory=list)
    column_selection: List[str] = Field(default_factory=list)
    filtering_and_sorting: List[str] = Field(default_factory=list)
    numeric_operations: List[str] = Field(default_factory=list)

class CommonErrorsAndCorrectionsInsights(BaseModel):
    schema_misunderstandings: List[str] = Field(default_factory=list)
    logical_errors: List[str] = Field(default_factory=list)
    query_structure_issues: List[str] = Field(default_factory=list)

class DataValidationTechniquesInsights(BaseModel):
    result_verification: List[str] = Field(default_factory=list)
    query_refinement: List[str] = Field(default_factory=list)

class SpecificDatabaseInsightsContent(BaseModel):
    # This will store insights for a specific database, e.g., "California Schools Database"
    # The key in the main model might be the database name, or we add a 'database_name' field here.
    # For simplicity in merging, let's assume a list of generic insights for now.
    # The markdown generation can handle specific database titles.
    database_name_hint: Optional[str] = Field(None, description="Hint for which database these insights pertain to, used for organizing markdown.")
    insights: List[str] = Field(default_factory=list, description="List of specific insights for this database.")

class GeneralSQLBestPracticesInsights(BaseModel):
    practices: List[str] = Field(default_factory=list, description="Numbered list of general SQL best practices observed or reinforced.")

class InsightsExtractionModel(BaseModel):
    """
    Structures the content for the cumulative summarized_insights.md file.
    This model will be populated by an LLM by processing individual feedback reports.
    """
    title: str = Field(default="# Summarized SQL Query Feedback and Insights", description="Main title of the insights document.")
    introduction: str = Field(
        default="This document summarizes key reasoning traces and insights from feedback on SQL queries. These insights can help improve SQL query generation and correction.",
        description="Introductory paragraph."
    )
    
    schema_understanding: SchemaUnderstandingInsights = Field(default_factory=SchemaUnderstandingInsights)
    query_construction_best_practices: QueryConstructionBestPracticesInsights = Field(default_factory=QueryConstructionBestPracticesInsights)
    common_errors_and_corrections: CommonErrorsAndCorrectionsInsights = Field(default_factory=CommonErrorsAndCorrectionsInsights)
    data_validation_techniques: DataValidationTechniquesInsights = Field(default_factory=DataValidationTechniquesInsights)
    
    # Using a dictionary to store insights for potentially multiple databases
    # Key: database_name_hint (e.g., "california_schools_db"), Value: List of insight strings
    specific_database_insights_map: Dict[str, List[str]] = Field(default_factory=dict, description="Map of database-specific insights. Key is a db identifier.")
    
    general_sql_best_practices: GeneralSQLBestPracticesInsights = Field(default_factory=GeneralSQLBestPracticesInsights)
    last_updated: Optional[str] = Field(None, description="Timestamp of the last update in ISO format.")

    def add_insight(self, section_path: str, insight_text: str, db_name_hint: Optional[str] = None):
        """
        Adds an insight to a specific section.
        For specific_database_insights, db_name_hint is required.
        Example section_paths:
        - "schema_understanding.table_structure_and_relationships"
        - "specific_database_insights_map.california_schools_db" (will add to list for this db)
        """
        parts = section_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if part == "specific_database_insights_map" and db_name_hint:
                if db_name_hint not in obj.specific_database_insights_map:
                    obj.specific_database_insights_map[db_name_hint] = []
                obj = obj.specific_database_insights_map # obj is now the dict
                # The next part (parts[-1]) should be the db_name_hint if we are adding to its list
                # This logic needs refinement if parts[-1] is not the list itself.
                # Let's simplify: if section_path starts with specific_database_insights_map,
                # the next part is the db_name_hint.
                break 
            obj = getattr(obj, part)

        target_attribute_name = parts[-1]

        if section_path.startswith("specific_database_insights_map."):
            if not db_name_hint:
                # db_name_hint should have been extracted from section_path or passed
                # For "specific_database_insights_map.california_schools_db", db_name_hint is "california_schools_db"
                # and target_attribute_name is also "california_schools_db" which is a key in the map.
                actual_db_name_hint = target_attribute_name 
            else:
                actual_db_name_hint = db_name_hint

            if actual_db_name_hint not in self.specific_database_insights_map:
                self.specific_database_insights_map[actual_db_name_hint] = []
            
            target_list = self.specific_database_insights_map[actual_db_name_hint]
            if insight_text not in target_list:
                target_list.append(insight_text)
                return True
        else:
            target_list = getattr(obj, target_attribute_name)
            if isinstance(target_list, list) and insight_text not in target_list:
                target_list.append(insight_text)
                return True
        return False

    def get_insights_for_db(self, db_name_hint: str) -> List[str]:
        return self.specific_database_insights_map.get(db_name_hint, [])

# Example of how InitializationResponse might look if we need it from initialization_module
# class InitializationResponse(BaseModel):
#     success: bool
#     message: str
#     db_name: Optional[str] = None
#     # schema_summary: Optional[str] = None # If returning the raw schema text
#     schema_and_sample_data: Optional[Dict[str, Any]] = None # For the new approach


# --- Models for Query Revision Feature ---

class SQLRevisionResponse(BaseModel):
    """
    Represents the structured response expected from the LLM for a single SQL revision iteration.
    """
    sql_query: Optional[str] = Field(None, description="The revised SQL query. Must start with SELECT.")
    explanation: Optional[str] = Field(None, description="Explanation of the revision or how the new query addresses the request.")
    error_message: Optional[str] = Field(None, description="If LLM itself detects an issue or cannot generate a revision.")

    @validator('sql_query', pre=True, always=True)
    def ensure_revised_sql_query_is_select(cls, value):
        if value and not value.strip().upper().startswith("SELECT"):
            # For now, let's assume the LLM will follow instructions, or handle non-SELECT upstream.
            # Pydantic validation here confirms adherence.
            pass 
        return value

class RevisionIteration(BaseModel):
    """
    Represents a single iteration within a query revision cycle.
    """
    user_revision_prompt: str = Field(description="The user's prompt for this revision iteration.")
    revised_sql_attempt: str = Field(description="The SQL query generated by the LLM for this iteration.")
    revised_explanation: str = Field(description="The LLM's explanation for the revised_sql_attempt.")

class NLQGenerationForRevisedSQLResponse(BaseModel):
    """
    Represents the LLM's response when generating a natural language question
    for a finalized revised SQL query.
    """
    natural_language_question: str = Field(description="The LLM-generated natural language question for the final SQL.")
    reasoning: Optional[str] = Field(None, description="The LLM's reasoning for generating this specific NLQ based on the revision history.")

class RevisionReportContentModel(BaseModel):
    """
    Structures the entire content and history of a query revision cycle.
    """
    initial_sql_for_revision: str = Field(description="The SQL query that was the starting point for revisions.")
    revision_iterations: List[RevisionIteration] = Field(default_factory=list, description="Chronological list of revision prompts and LLM attempts.")
    
    final_revised_sql_query: Optional[str] = Field(None, description="The very last SQL query after all revision iterations.")
    final_revised_explanation: Optional[str] = Field(None, description="The explanation for the final_revised_sql_query.")
    
    llm_generated_nlq_for_final_sql: Optional[str] = Field(None, description="The natural language question generated by the LLM for the final_revised_sql_query.")
    llm_reasoning_for_nlq: Optional[str] = Field(None, description="The LLM's reasoning for the generated NLQ.")
