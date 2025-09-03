import fastworkflow
import asyncio
from typing import Dict, Any
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.sql_generation import SQLGenerator

# Define the signature class for the generate_sql command
class Signature:
    class Input(BaseModel):
        # We use a single field to capture the entire natural language query
        # No parameter extraction - we want the complete query
        natural_language_question: str = Field(
            description="The complete natural language question to convert to SQL",
            examples=[
                "Show me all customers from California",
                "What are the top 10 products by sales?",
                "Find employees who have been with the company for more than 5 years",
                "List all accounts in the sales department",
                "Show me transactions over $1000 from last month"
            ],
            min_length=3,
            max_length=1000  # Increased to allow longer queries
        )

    # Plain utterances for training the intent detection model
    plain_utterances = [
        "Generate SQL to find all users who signed up last month",
        "Convert this question to SQL: how many orders were placed yesterday?",
        "Write a SQL query to show the top selling products",
        "Create a SQL query that lists all employees in the marketing department",
        "Turn this into SQL: which customers have spent more than $1000?",
        "Make an SQL query to find inactive accounts",
        "SQL for finding duplicate records in the customers table",
        "Generate a query to show sales by region",
        "Write SQL to join orders and customers tables",
        "Create a query that shows monthly revenue",
        "Generate sql to list employees",
        "List employees",
        "Show me employees",
        "Find employees",
        "Get employees",
        "Query employees",
        "SQL for employees",
        "Generate SQL to list 25 employees",
        "List 25 employees",
        "Show me 25 employees",
        "Find 25 employees",
        "Get 25 employees",
        "Query 25 employees",
        "SQL for 25 employees"
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
        
        # Add more SQL-specific utterances that don't explicitly mention "generate SQL"
        # These will help with intent detection without requiring the /generate_sql command
        implicit_sql_utterances = [
            "Convert this to SQL",
            "Turn this into a database query",
            "How would I query this from the database?",
            "What's the SQL for this question?",
            "Write a query for this",
            "Show me the SQL for",
            "Database query for this question",
            "SQL equivalent of this question",
            "How to query this from PostgreSQL",
            "Translate to SQL"
        ]
        
        return base_utterances + diverse_utterances + implicit_sql_utterances

# Define the response generator class for the generate_sql command
class ResponseGenerator:
    def __init__(self):
        self.sql_generator = None
    
    async def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Dict[str, Any]:
        """Helper function that executes the SQL generation logic"""
        # Check if the current state is SQLGenerationState
        current_state = getattr(workflow, "current_state", "InitializationState")
        if current_state != "SQLGenerationState":
            return {
                "sql_query": None, 
                "explanation": None, 
                "execution_result": None, 
                "execution_error": "Invalid state",
                "message_to_user": (
                    f"Error: Cannot generate SQL in the current state ({current_state}). "
                    f"You must initialize the database first before generating SQL."
                )
            }
        # Initialize the SQL generator if not already done
        if not self.sql_generator:
            # Get the database identifier from workflow.db_name_identifier or use default
            db_name_identifier = getattr(workflow, "db_name_identifier", getattr(workflow, "id", "default_db"))
            self.sql_generator = SQLGenerator(db_name_identifier)
        
        # Get the complete natural language question - no parameter extraction
        # We pass the entire query as is to the SQL generator
        nlq = input.natural_language_question
        print(f"Processing complete NLQ: {nlq}")
        
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
                "sql_query": None, 
                "explanation": None, 
                "execution_result": None, 
                "execution_error": "Database not initialized",
                "message_to_user": (
                    "Error: Database not initialized. Please run the initialize command first:\n\n"
                    "/initialize postgresql://postgres:postgresql@host.docker.internal:5432/demo_db\n\n"
                    "Or restart the application with the startup command parameter:\n\n"
                    "fastworkflow run /home/shivam/fastworkflow/postgres_copilot2 "
                    "/home/shivam/fastworkflow/postgres_copilot2/fastworkflow.env "
                    "/home/shivam/fastworkflow/postgres_copilot2/fastworkflow.passwords.env "
                    "--startup_command \"initialize postgresql://postgres:postgresql@host.docker.internal:5432/demo_db\""
                )
            }
        
        # Generate the SQL query using the complete natural language question
        result = await self.sql_generator.generate_sql_query(
            natural_language_question=nlq,
            schema_and_sample_data=schema_and_sample_data,
            insights_markdown_content=insights_markdown_content,
            llm_client=llm_client,
            row_limit_for_preview=1
        )
        
        # Initialize feedback iterations list
        workflow.feedback_iterations = []
        
        return result

    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """
        The framework will call this function to process the command.
        This method needs to be synchronous, so we'll use asyncio.run to run the async _process_command.
        """
        # Create a new event loop for this call
        loop = asyncio.new_event_loop()
        try:
            # Run the async _process_command in the new event loop
            result = loop.run_until_complete(self._process_command(workflow, command_parameters))
            
            # Extract the message to user from the result
            response = result.get("message_to_user", "Failed to generate SQL query.")
            
            # Print the natural language question and generated SQL for debugging
            print(f"Natural Language Question: {command_parameters.natural_language_question}")
            if result.get("sql_query"):
                print(f"Generated SQL: {result['sql_query']}")
                print(f"Explanation: {result.get('explanation', '')}")
                
                # Store the generated SQL, natural language question, and explanation in the workflow context
                # This will allow the approve_sql command to access this information
                workflow.current_sql = result["sql_query"]
                workflow.current_nlq = command_parameters.natural_language_question
                workflow.current_explanation = result.get("explanation", "")
                
                # Also store the original user query in the workflow context
                # This will be used by the approve_sql command to get the user's approval text
                workflow.last_user_query = command_parameters.natural_language_question
                
                # Set the workflow state to FeedbackApprovalState
                setattr(workflow, "current_state", "FeedbackApprovalState")
                
                # Set the context to FeedbackApprovalState
                if hasattr(workflow, "set_context"):
                    workflow.set_context("FeedbackApprovalState")
                
                # Add a note to the response about available commands
                response += "\n\nYou can now provide feedback on this SQL query or approve it."
            
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
