import fastworkflow
import asyncio
from typing import Dict, Any
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel

from ..application.sql_approval import SQLApprover

# Define the signature class for the approve_sql command
class Signature:
    # No input parameters needed as we're just detecting intent
    # We'll use the entire natural language prompt as is
    class Input(BaseModel):
        pass

    # Plain utterances for training the intent detection model
    plain_utterances = [
        "approve this sql",
        "approve the sql",
        "approve the query",
        "this sql looks good",
        "the query is correct",
        "save this sql",
        "save the query",
        "this query is good",
        "the sql is correct",
        "approve",
        "looks good",
        "that's correct",
        "that is right",
        "save this",
        "this is what I wanted",
        "perfect query",
        "excellent sql",
        "approve and save",
        "save and approve",
        "this works",
        "the query works",
        "the sql works",
        "approve and generate insights",
        "save this query and generate insights",
        "approve and save feedback",
        "save this for future reference",
        "add this to approved queries",
        "this is the right query",
        "this query is what I need",
        "approve this and save it"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        # Start with the command name itself as an utterance
        base_utterances = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ]
        
        # Add diverse variations of the plain utterances
        diverse_utterances = generate_diverse_utterances(Signature.plain_utterances, command_name)
        
        return base_utterances + diverse_utterances

# Define the response generator class for the approve_sql command
class ResponseGenerator:
    def __init__(self):
        self.sql_approver = None
    
    async def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input, user_query: str) -> Dict[str, Any]:
        """Helper function that executes the SQL approval logic"""
        # Check if the current state is FeedbackApprovalState
        current_state = getattr(workflow, "current_state", "InitializationState")
        if current_state != "FeedbackApprovalState":
            return {
                "success": False,
                "message": (
                    f"Error: Cannot approve SQL in the current state ({current_state}). "
                    f"You must first generate SQL using the generate_sql command before approving."
                )
            }
        # Initialize the SQL approver if not already done
        if not self.sql_approver:
            # Get the database identifier from workflow.db_name_identifier or use default
            db_name_identifier = getattr(workflow, "db_name_identifier", getattr(workflow, "id", "default_db"))
            self.sql_approver = SQLApprover(db_name_identifier)
        
        # Check if SQL has been generated
        if not hasattr(workflow, "current_sql") or not workflow.current_sql:
            return {
                "success": False,
                "message": "No SQL query to approve. Please generate SQL first using the generate_sql command."
            }
        
        if not hasattr(workflow, "current_nlq") or not workflow.current_nlq:
            return {
                "success": False,
                "message": "No natural language question found. Please generate SQL first using the generate_sql command."
            }
        
        # Get the current SQL and natural language question from the workflow context
        current_sql = workflow.current_sql
        current_nlq = workflow.current_nlq
        current_explanation = getattr(workflow, "current_explanation", "No explanation provided")
        
        # Get feedback iterations if they exist
        feedback_iterations = getattr(workflow, "feedback_iterations", [])
        
        # Get schema and insights from the workflow
        schema_and_sample_data = None
        insights_markdown_content = None
        
        # Try to get schema data from the workflow
        if hasattr(workflow, "schema_data"):
            schema_and_sample_data = workflow.schema_data
        elif hasattr(workflow, "get_schema_data"):
            schema_and_sample_data = workflow.get_schema_data()
        
        # Try to get insights from the workflow
        if hasattr(workflow, "insights_content"):
            insights_markdown_content = workflow.insights_content
        elif hasattr(workflow, "get_insights_content"):
            insights_markdown_content = workflow.get_insights_content()
        
        # Get the LLM client from the workflow if available
        llm_client = getattr(workflow, "llm_client", None)
        
        # Check if database has been initialized
        if not schema_and_sample_data or not llm_client:
            return {
                "success": False,
                "message": "Database not initialized. Please run the initialize command first."
            }
        
        # Process the approval
        result = await self.sql_approver.approve_sql(
            natural_language_question=current_nlq,
            sql_query=current_sql,
            explanation=current_explanation,
            user_approval_text=user_query,
            schema_and_sample_data=schema_and_sample_data,
            insights_markdown_content=insights_markdown_content,
            llm_client=llm_client,
            feedback_iterations=feedback_iterations  # Pass feedback iterations
        )
        
        # Reset the current SQL, NLQ, and feedback iterations in the workflow context
        # This ensures that the user needs to generate new SQL before approving again
        if result.get("success", False):
            workflow.current_sql = None
            workflow.current_nlq = None
            workflow.current_explanation = None
            workflow.feedback_iterations = []  # Reset feedback iterations
            
            # Set the workflow state back to SQLGenerationState
            setattr(workflow, "current_state", "SQLGenerationState")
            
            # Set the context to SQLGenerationState
            if hasattr(workflow, "set_context"):
                workflow.set_context("SQLGenerationState")
        
        return result

    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """
        The framework will call this function to process the command.
        This method needs to be synchronous, so we'll use asyncio.run to run the async _process_command.
        """
        # Get the original user query from the workflow context
        user_query = getattr(workflow, "last_user_query", "approve")
        
        # Create a new event loop for this call
        loop = asyncio.new_event_loop()
        try:
            # Run the async _process_command in the new event loop
            result = loop.run_until_complete(self._process_command(workflow, command_parameters, user_query))
            
            # Extract the message to user from the result
            response = result.get("message", "Failed to process SQL approval.")
            
            # If successful, add a note about being able to generate new SQL
            if result.get("success", False):
                response += "\n\nYou can now generate a new SQL query."
            
            # Return the command output
            return fastworkflow.CommandOutput(
                workflow_id=workflow.id,
                command_responses=[
                    fastworkflow.CommandResponse(response=response)
                ]
            )
        finally:
            # Close the event loop
            loop.close()
