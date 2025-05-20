import json
from typing import Tuple, Optional, Any, TYPE_CHECKING, Dict

# Assuming pydantic_models.py is in the same directory or accessible in PYTHONPATH
# from .pydantic_models import InitializationResponse # If you define a Pydantic model for the response

# Import memory_module
import memory_module

if TYPE_CHECKING:
    from postgres_copilot_chat import GeminiMcpClient # To avoid circular import

async def perform_initialization(
    mcp_client_session_handler: 'GeminiMcpClient', # Type hint with quotes for forward reference
    connection_string: str,
    db_name_identifier: Optional[str] = "default_db" # A name to identify this connection/summary
) -> Tuple[bool, str, Optional[Dict[str, Any]]]: # Returns: success_status, message_to_user, schema_data_dict
    """
    Connects to the PostgreSQL database using the MCP server, retrieves schema 
    and sample data, saves it using memory_module, and returns the data.

    Args:
        mcp_client_session_handler: The instance of GeminiMcpClient that holds the MCP session.
        connection_string: The PostgreSQL connection string.
        db_name_identifier: A unique name for this database connection (used for filenames).

    Returns:
        A tuple: (success_status, message_to_user, schema_and_sample_data_dict).
                 schema_and_sample_data_dict is None if initialization fails.
    """
    if not hasattr(mcp_client_session_handler, 'session') or not mcp_client_session_handler.session:
        return False, "Error: MCP session not available in the client handler.", None
    
    session = mcp_client_session_handler.session # Get the actual MCP session

    # --- Step 1: Connect to PostgreSQL via MCP tool ---
    print(f"Attempting to connect to PostgreSQL with db_name: {db_name_identifier}...")
    try:
        connect_result_obj = await session.call_tool(
            "connect_to_postgres", 
            {"connection_string": connection_string}
        )
        connect_message = mcp_client_session_handler._extract_mcp_tool_call_output(connect_result_obj)
        
        if not isinstance(connect_message, str) or "Error:" in connect_message or "Failed to connect" in connect_message :
            error_msg = connect_message if isinstance(connect_message, str) else "Unknown connection error."
            print(f"Connection failed: {error_msg}")
            return False, f"Failed to connect to database '{db_name_identifier}': {error_msg}", None
        print(f"Successfully connected to database: {db_name_identifier}. Message: {connect_message}")
    except Exception as e:
        print(f"Exception during connect_to_postgres tool call: {e}")
        return False, f"Exception while trying to connect to database '{db_name_identifier}': {e}", None

    # --- Step 2: Get Schema and Sample Data via MCP tool ---
    print("Attempting to fetch schema and sample data...")
    schema_data_dict: Optional[Dict[str, Any]] = None
    try:
        schema_data_result_obj = await session.call_tool("get_schema_and_sample_data", {})
        # _extract_mcp_tool_call_output should ideally return the dict directly if successful
        extracted_output = mcp_client_session_handler._extract_mcp_tool_call_output(schema_data_result_obj)

        if isinstance(extracted_output, dict):
            schema_data_dict = extracted_output
            if not schema_data_dict: # Empty dict might mean no user tables
                 print("Schema and sample data fetched, but the result is an empty dictionary (no user tables?).")
                 # This might be a valid state, so proceed to save an empty schema.
            else:
                print(f"Successfully fetched schema and sample data for {len(schema_data_dict)} tables.")
        elif isinstance(extracted_output, str):
            if "Error:" in extracted_output:
                print(f"Failed to get schema and sample data: {extracted_output}")
                return False, f"Connected, but failed to retrieve schema for '{db_name_identifier}': {extracted_output}", None
            elif "No user tables found" in extracted_output: # Handle specific string message for empty DB
                print(f"Schema and sample data: {extracted_output}")
                schema_data_dict = {} # Represent as empty schema
            else: # Try to parse the string as JSON
                try:
                    print(f"Received schema data as string, attempting to parse as JSON")
                    schema_data_dict = json.loads(extracted_output)
                    print(f"Successfully parsed schema data string as JSON with {len(schema_data_dict)} tables")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse schema data string as JSON: {e}")
                    return False, f"Connected, but received unexpected string response for schema that could not be parsed: {e}", None
        else:
            print(f"No schema data returned or unexpected format: {type(extracted_output)}")
            return False, "Connected, but no schema information was found or data format was unexpected.", None
            
    except Exception as e:
        print(f"Exception during get_schema_and_sample_data tool call: {e}")
        return False, f"Connected, but an exception occurred while fetching schema for '{db_name_identifier}': {e}", None

    # Ensure schema_data_dict is a dictionary before proceeding to save
    if schema_data_dict is None: # Should have been caught above, but as a safeguard
        print("Schema data dictionary is None before saving. This should not happen.")
        return False, "Internal error: Schema data became None before saving.", None

    # --- Step 3: Save Schema and Sample Data using memory_module ---
    try:
        # memory_module is already imported at the top level of this script
        schema_filepath = memory_module.save_schema_data(schema_data_dict, db_name_identifier)
        print(f"Schema and sample data saved to: {schema_filepath}")
    except Exception as e:
        print(f"Error saving schema and sample data using memory_module: {e}")
        # If saving fails, we might still want to proceed with the data in memory for the session
        # but inform the user. Or make it a fatal error for initialization.
        # For now, let's make it non-fatal for the session but report error.
        error_saving_msg = f"Connected and schema fetched for '{db_name_identifier}', but failed to save schema data locally: {e}. The data is available for this session."
        # Return True because we have the data, but the message reflects the save error.
        # The caller (postgres_copilot_chat) can decide how to handle this.
        # Or, to be stricter:
        # return False, f"Connected and schema fetched, but failed to save schema data locally for '{db_name_identifier}': {e}", None
        print(error_saving_msg) # Log it
        # Let's consider failure to save as a critical part of "initialization" for now.
        return False, f"Failed to save schema data for '{db_name_identifier}': {e}. Check permissions and paths.", None

    num_tables = len(schema_data_dict) if schema_data_dict is not None else 0
    success_message = (
        f"Successfully initialized database '{db_name_identifier}'. "
        f"Schema and sample data for {num_tables} tables fetched and saved. "
        f"Ready for SQL generation or navigation."
    )
    return True, success_message, schema_data_dict
