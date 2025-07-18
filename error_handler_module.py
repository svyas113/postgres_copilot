import sys
import re
from colorama import Fore, Style
import json
import os
import traceback
from datetime import datetime
import litellm
from pydantic import ValidationError


class SchemaTooLargeError(Exception):
    """Custom exception for when schema size exceeds the model's token limit."""
    def __init__(self, model_name, schema_tokens, available_tokens, message="Schema is too large for the model's context window."):
        self.model_name = model_name
        self.schema_tokens = schema_tokens
        self.available_tokens = available_tokens
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} Model: {self.model_name}, Schema Tokens: {self.schema_tokens}, Available Tokens: {self.available_tokens}"


LOGS_DIR = None
INFO_LOG_PATH = None
ERROR_LOG_PATH = None

def initialize_logs(memory_base_dir):
    """Set up the logging directory and log file paths."""
    global LOGS_DIR, INFO_LOG_PATH, ERROR_LOG_PATH
    LOGS_DIR = os.path.join(memory_base_dir, 'logs')
    os.makedirs(LOGS_DIR, exist_ok=True)
    INFO_LOG_PATH = os.path.join(LOGS_DIR, 'info.log')
    ERROR_LOG_PATH = os.path.join(LOGS_DIR, 'error.log')

def display_message(message, level="INFO", log=True):
    """
    Displays a message to the user with appropriate coloring and logging.
    Levels: INFO, WARNING, ERROR, FATAL.
    """
    level = level.upper()
    log_message = f"[{datetime.utcnow().isoformat()}] [{level}] {message}\n"

    if level == "INFO":
        print(f"{Fore.CYAN}{message}{Style.RESET_ALL}", file=sys.stdout)
        if log and INFO_LOG_PATH:
            try:
                with open(INFO_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(log_message)
            except Exception as e:
                print(f"{Fore.RED}Critical: Failed to write to info log: {e}{Style.RESET_ALL}", file=sys.stderr)

    elif level == "WARNING":
        print(f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}", file=sys.stderr)
        if log and ERROR_LOG_PATH:
            try:
                with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(log_message)
            except Exception as e:
                print(f"{Fore.RED}Critical: Failed to write to error log: {e}{Style.RESET_ALL}", file=sys.stderr)

    elif level in ["ERROR", "FATAL"]:
        print(f"{Fore.RED}Error: {message}{Style.RESET_ALL}", file=sys.stderr)
        if log and ERROR_LOG_PATH:
            try:
                with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(log_message)
            except Exception as e:
                print(f"{Fore.RED}Critical: Failed to write to error log: {e}{Style.RESET_ALL}", file=sys.stderr)
        
        if level == "FATAL":
            print(f"{Fore.RED}This is a fatal error. Exiting.{Style.RESET_ALL}", file=sys.stderr)
            sys.exit(1)

def log_exception(exc, user_query=None, context=None):
    """Logs a structured exception to a JSON file."""
    if not LOGS_DIR:
        display_message("Logging directory not initialized. Call initialize_logs first.", "ERROR")
        return

    error_log_path = os.path.join(LOGS_DIR, 'error_log.json')
    
    tb_list = traceback.extract_tb(exc.__traceback__)
    formatted_tb = [{
        "file": item.filename,
        "line": item.lineno,
        "function": item.name,
        "code": item.line
    } for item in tb_list]

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_query": user_query,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": formatted_tb,
        "context": context or {}
    }

    try:
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        display_message(f"Critical: Failed to write to structured error log file: {e}", "ERROR")

def handle_exception(exc, user_query=None, context=None) -> str:
    """
    Logs an exception and returns a user-friendly error message string.
    """
    log_exception(exc, user_query, context)

    if isinstance(exc, SchemaTooLargeError):
        message = (
            f"The schema for the requested tables is too large for the current model ({exc.model_name}).\n"
            f"Schema tokens: {exc.schema_tokens}, Available: {exc.available_tokens}\n"
            "Please edit the active_tables.txt file to reduce the number of tables and then run /reload_scope."
        )
    elif isinstance(exc, litellm.ContextWindowExceededError):
        # Extract context window and prompt tokens from the error message if available
        error_str = str(exc)
        # Example message: "litellm.ContextWindowExceededError: litellm.BadRequestError: Azure OpenAI Error - {'error': {'message': 'The prompt is too long. Your prompt has 10000 tokens. The maximum token limit for this model is 8192 tokens. ...
        match = re.search(r"Your prompt has (\d+) tokens\. The maximum token limit for this model is (\d+) tokens", error_str)
        if match:
            prompt_tokens = int(match.group(1))
            limit_tokens = int(match.group(2))
            over_by = prompt_tokens - limit_tokens
            token_info = f" (Your prompt: {prompt_tokens} tokens, Model limit: {limit_tokens}, Over by: {over_by} tokens)"
        else:
            token_info = ""

        db_name = context.get('db_name_identifier', 'your_db') if context else 'your_db'
        
        message = (
            f"The model's context window was exceeded{token_info}. The request is too large.\n"
            f"This usually happens when the database schema is very large.\n\n"
            f"{Fore.YELLOW}ACTION: To fix this, please edit the scope of the database.\n"
            f"1. Open this file in a text editor: {Fore.WHITE}data/memory/{db_name}/active_tables.txt{Style.RESET_ALL}\n"
            f"{Fore.YELLOW}2. Remove the lines for any tables that are not relevant to your query.\n"
            f"3. Save the file and then run the command: {Fore.WHITE}/reload_scope{Style.RESET_ALL}"
        )
    elif isinstance(exc, litellm.AuthenticationError):
        message = (
            "LLM API authentication failed. Please check your API key, account credits, and model access."
        )
    elif isinstance(exc, litellm.NotFoundError):
        message = "The requested LLM model was not found. Please check the model name in your profile."
    elif isinstance(exc, litellm.RateLimitError):
        message = "The LLM API rate limit has been exceeded. Please wait and try again."
    elif isinstance(exc, litellm.APIError) and exc.status_code and exc.status_code >= 500:
        message = f"The LLM service reported a server error (HTTP {exc.status_code}). This is an issue with the provider, not your request. Please try again later."
    elif isinstance(exc, (ValidationError, json.JSONDecodeError)):
        message = "The AI's response was not in the expected format. Please try rephrasing your query."
    elif 'connection' in str(exc).lower():
        message = "A network connection error occurred. Please check your internet and service status."
    else:
        message = (
            "An unexpected error occurred. Technical details have been logged. "
            "Try rephrasing your query or restarting."
        )
    
    return message

def display_response(message):
    """Displays a standard AI response to the user."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}AI:{Style.RESET_ALL} {message}")

def log_mcp_response(tool_name, response):
    """Logs a complete MCP server response to a JSON file."""
    if not LOGS_DIR:
        print("Error: Logging directory not initialized. Call initialize_logs first.")
        return

    mcp_log_path = os.path.join(LOGS_DIR, 'mcp_server.log')
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool_name": tool_name,
        "response": response
    }

    try:
        with open(mcp_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Critical: Failed to write to MCP log file: {e}")
