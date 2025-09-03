import fastworkflow
import os
import sys
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.initialization import perform_initialization
from ..application.db_session import PostgreSQLSession
from ..application.llm_client import create_llm_client

# The signature class defines our intent
class Signature:
    class Input(BaseModel):
        connection_string: str = Field(
            description="PostgreSQL connection string (optional, will use environment variable if not provided)",
            examples=[
                "postgresql://user:password@localhost:5432/dbname",
                "postgresql://postgres:postgres@localhost:5432/postgres"
            ],
            pattern=r'^(postgresql://[^:]+:[^@]+@[^:]+:\d+/[^/]+)?$',  # Made pattern optional with ? at the end
            default=""
        )
        db_name_identifier: str = Field(
            description="Identifier for the database (optional, will use database name from connection string if not provided)",
            examples=["codebase_community", "financial_db", "ecommerce_db"],
            default=""
        )
        force_regenerate: bool = Field(
            description="If True, regenerate schema vectors and graph even if they exist",
            default=False
        )

    class Output(BaseModel):
        success: bool = Field(description="Whether the initialization was successful")
        message: str = Field(description="Message describing the result of the initialization")

    plain_utterances = [
        "Connect to postgresql://user:password@localhost:5432/dbname",
        "Initialize database postgresql://postgres:postgres@localhost:5432/postgres",
        "Connect to PostgreSQL database at localhost:5432",
        "Set up connection to my database",
        "Initialize the database with connection string",
        "Connect to my PostgreSQL server",
        "Set up database connection",
        "Initialize database with force regenerate",
        "Connect and regenerate schema vectors",
        "Initialize",
        "Initialize database",
        "Initialize postgres",
        "Connect to database",
        "Setup database connection"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

# The response generator class processes the command
class ResponseGenerator:
    async def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Signature.Output:
        """Helper function that actually executes the initialization."""
        
        # Get connection string from input or environment variable
        connection_string = input.connection_string
        if not connection_string:
            # Use the connection string from the environment variable
            connection_string = os.environ.get("DATABASE_CONNECTION_STRING", "")
            
            # If still not found, try to read directly from the env file
            if not connection_string:
                try:
                    # First try the current directory
                    env_file_paths = [
                        "/home/shivam/fastworkflow/postgres_copilot (new)/fastworkflow.env",
                        "/home/shivam/fastworkflow/postgres_copilot3/fastworkflow.env",
                        "./fastworkflow.env",
                        "../fastworkflow.env"
                    ]
                    
                    for env_file_path in env_file_paths:
                        if os.path.exists(env_file_path):
                            print(f"[INFO] Reading environment from {env_file_path}")
                            with open(env_file_path, 'r') as f:
                                for line in f:
                                    if line.strip() and not line.strip().startswith('#'):
                                        try:
                                            key, value = line.strip().split('=', 1)
                                            if key == "DATABASE_CONNECTION_STRING":
                                                connection_string = value
                                                print(f"[INFO] Found DATABASE_CONNECTION_STRING in env file")
                                                break
                                        except ValueError:
                                            # Skip lines that don't have a key=value format
                                            continue
                            
                            # If we found the connection string, break out of the loop
                            if connection_string:
                                break
                except Exception as e:
                    print(f"[INFO] Error reading env file: {e}")
            
            if not connection_string:
                return Signature.Output(
                    success=False,
                    message="No connection string provided and DATABASE_CONNECTION_STRING environment variable is not set."
                )
        
        # Extract database name from connection string
        # Format: postgresql://user:password@host:port/dbname
        try:
            db_name_from_connection = connection_string.split('/')[-1]
        except Exception:
            db_name_from_connection = "default_db"
        
        # If db_name_identifier is not provided, use the database name from the connection string
        db_name_identifier = input.db_name_identifier
        if not db_name_identifier:
            db_name_identifier = db_name_from_connection
        
        # Check if we need to use existing schema files for a different db_name_identifier
        # This happens when the db_name_identifier is different from the database name in the connection string
        # and we want to use existing schema files
        if db_name_identifier != db_name_from_connection:
            print(f"[INFO] Using db_name_identifier '{db_name_identifier}' with database '{db_name_from_connection}' from connection string")
        
        # Create a database session
        db_session = PostgreSQLSession(connection_string)
        
        # Test the database connection
        if not db_session.is_connected():
            return Signature.Output(
                success=False,
                message="Failed to connect to the database. Please check your connection string."
            )
        
        # Create an LLM client with the database session
        llm_client = create_llm_client(session=db_session)
        
        # Call the initialization function
        success, message, schema_data = await perform_initialization(
            connection_string=connection_string,
            db_name_identifier=db_name_identifier,
            force_regenerate=input.force_regenerate
        )
        
        # If successful, update the workflow context with the schema data, database name, and LLM client
        if success and schema_data:
            # Check if schema_data is a dictionary with schema_data key
            if isinstance(schema_data, dict) and "schema_data" in schema_data:
                # Set schema_data attribute directly on the workflow object
                setattr(workflow, "schema_data", schema_data["schema_data"])
            else:
                # Fallback to the old behavior
                setattr(workflow, "schema_data", schema_data)
            
            # Set the LLM client with the database session
            setattr(workflow, "llm_client", llm_client)
            
            # Set the database session directly on the workflow object
            setattr(workflow, "db_session", db_session)
                
            # Set other attributes
            setattr(workflow, "current_database", db_name_identifier)
            setattr(workflow, "connection_string", connection_string)
            setattr(workflow, "db_name_identifier", db_name_identifier)
            
            # Load insights content if available
            # In a real implementation, we would load insights from a file
            # For now, we'll just set an empty string
            setattr(workflow, "insights_content", "")
            
            # Set the workflow state to SQLGenerationState
            setattr(workflow, "current_state", "SQLGenerationState")
            
            # Set the context to SQLGenerationState
            if hasattr(workflow, "set_context"):
                workflow.set_context("SQLGenerationState")
        
        return Signature.Output(
            success=success,
            message=message
        )

    def __call__(self, workflow: fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        
        try:
            # Process the initialization - use asyncio.run to handle the coroutine
            import asyncio
            output = asyncio.run(self._process_command(workflow, command_parameters))
            
            # Format the response for the user
            if output.success:
                # Set the initial state if not already set
                if not hasattr(workflow, "current_state"):
                    setattr(workflow, "current_state", "SQLGenerationState")
                
                # Set the context to SQLGenerationState
                if hasattr(workflow, "set_context"):
                    workflow.set_context("SQLGenerationState")
                
                response_text = f"✅ Database initialization successful!\n\n{output.message}\n\nYou can now generate SQL queries using natural language."
            else:
                # Set the state to InitializationState if initialization failed
                setattr(workflow, "current_state", "InitializationState")
                
                # Set the context to InitializationState
                if hasattr(workflow, "set_context"):
                    workflow.set_context("InitializationState")
                
                response_text = f"❌ Database initialization failed.\n\n{output.message}"

            return fastworkflow.CommandOutput(
                workflow_id=workflow.id,
                command_responses=[
                    fastworkflow.CommandResponse(response=response_text)
                ]
            )
            
        except Exception as e:
            error_message = f"Error during database initialization: {str(e)}"
            return fastworkflow.CommandOutput(
                workflow_id=workflow.id,
                command_responses=[
                    fastworkflow.CommandResponse(response=error_message)
                ]
            )
