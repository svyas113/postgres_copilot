import fastworkflow
import asyncio
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Import with proper fallback pattern
try:
    from ..application.schema_navigation_module import SchemaNavigationModule
    from ..application.llm_client import create_llm_client
except ImportError:
    try:
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from application.schema_navigation_module import SchemaNavigationModule
        from application.llm_client import create_llm_client
    except ImportError:
        print("Error importing required modules for schema_navigation command")
        raise

class Signature:
    class Input(BaseModel):
        question: str = Field(
            description="Natural language question about the database schema",
            examples=[
                "Tell me about this database", 
                "What does the users table contain?",
                "What are the relationships between the orders and customers tables?",
                "Summarize the database schema"
            ],
            min_length=3
        )

    plain_utterances = [
        "Tell me about this database",
        "What tables are in this database?",
        "Explain the schema of this database",
        "What does the users table contain?",
        "Describe the structure of the orders table",
        "What are the relationships between tables?",
        "How are customers and orders related?",
        "What columns are in the products table?",
        "Summarize this database schema for me",
        "What primary keys are used in this database?"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    async def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Dict[str, Any]:
        """Helper function that executes the schema navigation logic"""
        # Get the question from input parameters
        question = input.question
        
        # Get the current database identifier from the workflow context
        db_name_identifier = getattr(workflow, "current_database", None)
        if not db_name_identifier:
            return {
                "answer": "Error: No database has been initialized. Please initialize a database first.",
                "success": False
            }
        
        # Create LLM client
        llm_client = create_llm_client(getattr(workflow, "session", None))
        
        # Get previous answer from workflow context if available
        previous_answer = getattr(workflow, "previous_schema_answer", None)
        
        # Create schema navigation module
        schema_nav = SchemaNavigationModule(db_name_identifier)
        
        # Get the answer
        answer = await schema_nav.answer_schema_question(question, llm_client, previous_answer)
        
        # Store the answer in workflow context for future follow-up questions
        setattr(workflow, "previous_schema_answer", answer)
        
        return {
            "answer": answer,
            "success": True
        }
    
    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """
        The framework will call this function to process the command.
        This method needs to be synchronous, so we'll use asyncio.run_until_complete to run the async _process_command.
        """
        # Create a new event loop for this call
        loop = asyncio.new_event_loop()
        try:
            # Run the async _process_command in the new event loop
            result = loop.run_until_complete(self._process_command(workflow, command_parameters))
            
            # Extract the answer from the result
            response = result.get("answer", "Failed to answer the question about the database schema.")
            
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
