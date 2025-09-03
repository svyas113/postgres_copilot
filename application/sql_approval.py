import asyncio
import json
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

# Import necessary modules
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import application modules
from . import memory_module
from . import insights_module
from .pydantic_models import FeedbackReportContentModel, FeedbackIteration

class SQLApprover:
    """
    Handles the approval of SQL queries, including generating feedback reports,
    updating insights, and saving approved SQL pairs.
    """
    
    def __init__(self, db_name_identifier: str):
        """
        Initialize the SQL approver with the database identifier.
        
        Args:
            db_name_identifier: The identifier for the database.
        """
        self.db_name_identifier = db_name_identifier
    
    async def approve_sql(
        self,
        natural_language_question: str,
        sql_query: str,
        explanation: str,
        user_approval_text: str,
        schema_and_sample_data: Dict[str, Any],
        insights_markdown_content: Optional[str],
        llm_client: Any,
        feedback_iterations: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process the approval of a SQL query.
        
        Args:
            natural_language_question: The original natural language question.
            sql_query: The SQL query to approve.
            explanation: The explanation for the SQL query.
            user_approval_text: The user's approval text.
            schema_and_sample_data: The database schema and sample data.
            insights_markdown_content: The current insights markdown content.
            llm_client: The LLM client for generating feedback and insights.
            
        Returns:
            A dictionary with the result of the approval process.
        """
        try:
            # Create a feedback report
            feedback_report = self._create_feedback_report(
                natural_language_question=natural_language_question,
                sql_query=sql_query,
                explanation=explanation,
                user_approval_text=user_approval_text,
                feedback_iterations=feedback_iterations
            )
            
            # Save the feedback report
            feedback_filepath = memory_module.save_feedback_markdown(
                feedback_report, 
                self.db_name_identifier
            )
            
            # Read back the saved feedback report
            saved_feedback_md_content = memory_module.read_feedback_file(feedback_filepath)
            if not saved_feedback_md_content:
                return {
                    "success": False,
                    "message": "Could not read back saved feedback file for insights processing."
                }
            
            # Generate and update insights
            insights_success = await insights_module.generate_and_update_insights(
                llm_client, 
                saved_feedback_md_content,
                self.db_name_identifier,
                sql_query=sql_query,
                user_feedback=user_approval_text
            )
            
            # Save the NL2SQL pair
            try:
                memory_module.save_nl2sql_pair(
                    self.db_name_identifier,
                    natural_language_question,
                    sql_query
                )
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to save NL2SQL pair: {e}"
                }
            
            # Construct the success message
            message = f"SQL query approved and saved. Feedback report and NL2SQL pair for '{self.db_name_identifier}' saved."
            if insights_success:
                message += " Insights successfully generated/updated."
            else:
                message += " Failed to generate or update insights."
            
            return {
                "success": True,
                "message": message,
                "feedback_filepath": str(feedback_filepath)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during approval process: {e}"
            }
    
    def _create_feedback_report(
        self,
        natural_language_question: str,
        sql_query: str,
        explanation: str,
        user_approval_text: str,
        feedback_iterations: List[Dict[str, str]] = None
    ) -> FeedbackReportContentModel:
        """
        Create a feedback report for the approved SQL query.
        
        Args:
            natural_language_question: The original natural language question.
            sql_query: The SQL query to approve.
            explanation: The explanation for the SQL query.
            user_approval_text: The user's approval text.
            feedback_iterations: List of feedback iterations.
            
        Returns:
            A FeedbackReportContentModel instance.
        """
        # Convert feedback iterations to FeedbackIteration objects
        feedback_iteration_objects = []
        
        # The current SQL and explanation are the final ones after all feedback
        # We'll use them as the initial values if there are no feedback iterations
        initial_sql = sql_query
        initial_explanation = explanation
        
        if feedback_iterations and len(feedback_iterations) > 0:
            # If there are feedback iterations, we need to determine the initial SQL and explanation
            # The initial SQL and explanation are what we had before any feedback was applied
            # We don't have direct access to this, but we can infer it from the workflow context
            
            # For now, we'll use the current SQL and explanation as both initial and final
            # In a more complete implementation, we would store the original SQL and explanation
            # before any feedback was applied
            
            # Convert feedback iterations to FeedbackIteration objects
            for iteration in feedback_iterations:
                feedback_iteration_objects.append(FeedbackIteration(
                    user_feedback_text=iteration["user_feedback_text"],
                    corrected_sql_attempt=iteration["corrected_sql_attempt"],
                    corrected_explanation=iteration["corrected_explanation"]
                ))
        
        # Create a feedback report with the approved SQL
        feedback_report = FeedbackReportContentModel(
            natural_language_question=natural_language_question,
            initial_sql_query=initial_sql,
            initial_explanation=initial_explanation,
            feedback_iterations=feedback_iteration_objects,
            final_corrected_sql_query=sql_query,
            final_explanation=explanation,
            why_initial_query_was_wrong_or_suboptimal=(
                "The initial query was correct and approved by the user." 
                if not feedback_iterations else 
                "The initial query required revisions based on user feedback."
            ),
            why_final_query_works_or_is_improved=(
                f"The query was approved by the user with: '{user_approval_text}'"
                if not feedback_iterations else
                f"After {len(feedback_iterations)} rounds of feedback, the query was approved by the user with: '{user_approval_text}'"
            ),
            database_insights_learned_from_this_query=[
                f"This query was approved for the question: '{natural_language_question}'"
            ],
            sql_lessons_learned_from_this_query=[
                "User-approved queries provide valuable examples for future SQL generation."
            ]
        )
        
        return feedback_report
