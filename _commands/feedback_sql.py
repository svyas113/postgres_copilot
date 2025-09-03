import fastworkflow
import asyncio
from typing import Dict, Any
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.sql_feedback import SQLFeedbackProcessor

# Define the signature class for the feedback_sql command
class Signature:
    # We need to capture the feedback text as a parameter
    class Input(BaseModel):
        feedback_text: str = Field(
            description="The feedback text provided by the user",
            examples=[
                "revise query to list only the employees of Executive board department",
                "add a WHERE clause to filter by department name",
                "change the ORDER BY clause to sort by last name"
            ],
            min_length=3,
            max_length=1000
        )

    # Plain utterances for training the intent detection model
    plain_utterances = [
        # Generic feedback phrases
        "feedback on the sql",
        "revise the query",
        "update the sql",
        "change the query",
        "modify the sql",
        "improve the query",
        "fix the sql",
        "the query should",
        "the sql should",
        
        # Specific feedback examples with complete sentences
        "revise this query to use Sales instead of Pricing in the where clause",
        "update the sql to join with the Customers table",
        "change the query to filter by department name",
        "modify the sql to include employee hire date",
        "the query should order results by last name",
        "add a limit of 10 to the query",
        "filter the results to only show active employees",
        "group the results by department instead of location",
        "exclude terminated employees from the query",
        "add a where clause to filter by date",
        "change the select columns to include email address",
        "the query needs to calculate the average salary",
        "revise the query to show only Houston factory employees",
        "modify the sql to use a left join instead",
        "update the query to use the correct table name",
        "the sql should use a subquery for the department filter",
        "change the order by clause to sort by descending order",
        "revise the query to use a case statement for the status",
        "update the sql to include a having clause",
        "modify the query to use a different join condition",
        
        # Action-oriented feedback phrases
        "add to the query",
        "include in the sql",
        "exclude from the query",
        "sort the results by",
        "filter the results by",
        "group the results by",
        "join with another table",
        "limit the results",
        "the query needs to",
        "the sql needs to",
        "can you change the query to",
        "please update the sql to",
        "revise the query to show",
        "modify the sql to include",
        "change the query to exclude",
        "update the sql to sort by",
        "revise the query to filter by",
        "modify the sql to group by",
        "change the query to join with",
        "update the sql to limit",
        "revise the query to calculate"
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

# Define the response generator class for the feedback_sql command
class ResponseGenerator:
    def __init__(self):
        self.sql_feedback_processor = None
    
    async def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input, user_query: str) -> Dict[str, Any]:
        """Helper function that executes the SQL feedback logic"""
        # Check if the current state is FeedbackApprovalState
        current_state = getattr(workflow, "current_state", "InitializationState")
        if current_state != "FeedbackApprovalState":
            return {
                "success": False,
                "message": (
                    f"Error: Cannot provide feedback in the current state ({current_state}). "
                    f"You must first generate SQL using the generate_sql command before providing feedback."
                )
            }
        # Initialize the SQL feedback processor if not already done
        if not self.sql_feedback_processor:
            # Get the database identifier from workflow.db_name_identifier or use default
            db_name_identifier = getattr(workflow, "db_name_identifier", getattr(workflow, "id", "default_db"))
            self.sql_feedback_processor = SQLFeedbackProcessor(db_name_identifier)
        
        # Check if SQL has been generated
        if not hasattr(workflow, "current_sql") or not workflow.current_sql:
            return {
                "success": False,
                "message": "No SQL query to provide feedback on. Please generate SQL first using the generate_sql command."
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
        
        # Get feedback iterations if they exist, or create a new list
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
        
        # Process the feedback
        result = await self.sql_feedback_processor.process_feedback(
            natural_language_question=current_nlq,
            current_sql=current_sql,
            current_explanation=current_explanation,
            feedback_text=user_query,
            feedback_iterations=feedback_iterations,
            schema_and_sample_data=schema_and_sample_data,
            insights_markdown_content=insights_markdown_content,
            llm_client=llm_client
        )
        
        # Update the workflow context with the new SQL, explanation, and execution results
        if result.get("success", False):
            workflow.current_sql = result.get("revised_sql")
            workflow.current_explanation = result.get("revised_explanation")
            
            # Store execution results in the workflow context
            workflow.execution_result = result.get("execution_result")
            workflow.execution_error = result.get("execution_error")
            
            # Add the new feedback iteration to the list
            new_iteration = {
                "user_feedback_text": user_query,
                "corrected_sql_attempt": result.get("revised_sql"),
                "corrected_explanation": result.get("revised_explanation"),
                "execution_result": result.get("execution_result"),
                "execution_error": result.get("execution_error")
            }
            feedback_iterations.append(new_iteration)
            
            # Update the feedback iterations in the workflow context
            workflow.feedback_iterations = feedback_iterations
        
        return result

    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """
        The framework will call this function to process the command.
        This method needs to be synchronous, so we'll use asyncio.run to run the async _process_command.
        """
        # Get the feedback text directly from the command parameters
        user_query = command_parameters.feedback_text
        
        # Store the feedback text in the workflow context for future reference
        workflow.last_user_query = user_query
        
        print(f"Processing feedback: {user_query}")
        
        # Create a new event loop for this call
        loop = asyncio.new_event_loop()
        try:
            # Run the async _process_command in the new event loop
            result = loop.run_until_complete(self._process_command(workflow, command_parameters, user_query))
            
            # Extract the message to user from the result
            response = result.get("message", "Failed to process SQL feedback.")
            
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
