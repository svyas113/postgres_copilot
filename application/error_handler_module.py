import sys
from typing import Optional, Dict, Any

def handle_exception(e: Exception, user_query: Optional[str] = None, context: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> str:
    """
    Enhanced error handling function with better logging.
    
    Args:
        e: The exception that was raised
        user_query: The user query that caused the exception, if applicable
        context: Additional context information as a dictionary
        message: A user-friendly message to display
        
    Returns:
        The error message as a string
    """
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
    
    # Print user-friendly message if provided
    if message:
        print(f"[INFO] {message}")
    
    # Return the error message for the caller
    return str(e)

def display_message(message: str, level: str = "INFO", log: bool = True) -> None:
    """
    Display a message to the user with a specified level.
    
    Args:
        message: The message to display
        level: The level of the message (INFO, WARNING, ERROR, FATAL)
        log: Whether to log the message to stderr (for ERROR and FATAL)
    """
    level_upper = level.upper()
    
    # Print the message with the appropriate level
    print(f"[{level_upper}] {message}")
    
    # Log errors and fatal messages to stderr
    if log and level_upper in ["ERROR", "FATAL"]:
        print(f"[{level_upper}] {message}", file=sys.stderr)

def display_response(message: str) -> None:
    """
    Display a response message to the user without any level prefix.
    
    Args:
        message: The message to display
    """
    print(message)

def log_mcp_response(tool_name: str, output: Any) -> None:
    """
    Log an MCP tool response for debugging purposes.
    
    Args:
        tool_name: The name of the MCP tool
        output: The output from the tool
    """
    # This is a no-op in production, but can be enabled for debugging
    # print(f"MCP Tool '{tool_name}' response: {output}", file=sys.stderr)
    pass
