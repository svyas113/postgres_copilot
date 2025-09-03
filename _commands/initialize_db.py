import fastworkflow
import os
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.initialization import perform_initialization
from ..application.db_session import PostgreSQLSession
from ..application.llm_client import create_llm_client

# The signature class defines our intent
class Signature:
    class Input(BaseModel):
        # No parameters required for this command
        pass

    class Output(BaseModel):
        success: bool = Field(description="Whether the initialization was successful")
        message: str = Field(description="Message describing the result of the initialization")

    plain_utterances = [
        "Initialize",
        "Initialize database",
        "Initialize postgres",
        "Connect to database",
        "Setup database connection",
        "Connect to postgres",
        "Setup postgres",
        "Start database",
        "Connect to the database"
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
        
        # Read the connection string from the environment variable
        connection_string = os.environ.get("DATABASE_CONNECTION_STRING", "")
        
        # If not found in environment variables, try to read directly from the env file
        if not connection_string:
            try:
                # Try different possible locations for the env file
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
                message="No connection string found in environment variables or env file."
            )
        
        # Extract database name from connection string
        # Format: postgresql://user:password@host:port/dbname
        try:
            db_name_from_connection = connection_string.split('/')[-1]
        except Exception:
            db_name_from_connection = "default_db"
        
        # Use the database name from the connection string as the db_name_identifier
        db_name_identifier = db_name_from_connection
        
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
            force_regenerate=False
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
                
            # Set other attributes
            setattr(workflow, "current_database", db_name_identifier)
            setattr(workflow, "connection_string", connection_string)
            setattr(workflow, "db_name_identifier", db_name_identifier)
            setattr(workflow, "db_session", db_session)
            
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
