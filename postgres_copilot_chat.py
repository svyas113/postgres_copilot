import asyncio
import os
import sys
import subprocess # Keep for StdioServerParameters if server is run as subprocess
from dotenv import load_dotenv
from typing import Optional, Any, Dict, Tuple
from contextlib import AsyncExitStack
import datetime
import litellm
from litellm import acompletion # For asynchronous calls
from pydantic import ValidationError # For catching Pydantic errors
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool 
import copy
import re
import json
import memory_module
import initialization_module
import sql_generation_module
import insights_module
import database_navigation_module
import revise_query_module # Added
import vector_store_module # Added for RAG
from colorama import Fore, Style, init as colorama_init # Added for Colorama
# Removed: from config.settings import app_config, vector_store_config
from pydantic_models import ( # Updated imports
    SQLGenerationResponse, 
    FeedbackReportContentModel, 
    FeedbackIteration,
    RevisionReportContentModel, # Added
    RevisionIteration # Added
)

# --- Directory and .env Setup ---
# These setup steps are now primarily handled by prerequisites.py
# We still need to load .env here for the application to use the API key.
PASSWORDS_DIR = os.path.join(os.path.dirname(__file__), 'passwords') # Corrected for script within db-copilot
dotenv_path = os.path.join(PASSWORDS_DIR, '.env')

def check_available_providers():
    """Check which LLM providers have credentials available."""
    providers = []
    
    # Check Google AI
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    # Debug print for API keys
    print(f"DEBUG: (db-copilot script) GEMINI_API_KEY from env: {'SET' if gemini_key else 'NOT SET'}", file=sys.stderr)
    if gemini_key:
        print(f"DEBUG: (db-copilot script) GEMINI_API_KEY length: {len(gemini_key)}, first 5 chars: {gemini_key[:5] if gemini_key else ''}", file=sys.stderr)
    print(f"DEBUG: (db-copilot script) GOOGLE_API_KEY from env: {'SET' if google_key else 'NOT SET'}", file=sys.stderr)
    if google_key:
        print(f"DEBUG: (db-copilot script) GOOGLE_API_KEY length: {len(google_key)}, first 5 chars: {google_key[:5] if google_key else ''}", file=sys.stderr)
        
    if gemini_key or google_key:
        providers.append("Google AI (Gemini)")
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    
    # Check AWS Bedrock
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        providers.append("AWS Bedrock")
    
    # Check Anthropic (direct)
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("Anthropic")
    
    return providers

if os.path.exists(dotenv_path):
    print(f"DEBUG: (db-copilot script) Loading .env file from: {dotenv_path}", file=sys.stderr)
    dotenv_loaded = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
    print(f"DEBUG: (db-copilot script) python-dotenv load_dotenv result: {dotenv_loaded}", file=sys.stderr)
    
    # Immediately check a known key after loading
    debug_gemini_key_after_load = os.getenv("GEMINI_API_KEY")
    print(f"DEBUG: (db-copilot script) GEMINI_API_KEY immediately after load_dotenv: {'SET' if debug_gemini_key_after_load else 'NOT SET'}", file=sys.stderr)
    if debug_gemini_key_after_load:
         print(f"DEBUG: (db-copilot script) GEMINI_API_KEY length after load: {len(debug_gemini_key_after_load)}, first 5: {debug_gemini_key_after_load[:5]}", file=sys.stderr)

    available_providers = check_available_providers()
    
    if not available_providers:
        print(f"Warning: No API keys for common LLM providers found in {dotenv_path}.")
        print("Please ensure the correct keys for your chosen LiteLLM model are set.")
    else:
        print(f"Available LLM providers: {', '.join(available_providers)}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. API keys for LiteLLM will likely be missing. Please run prerequisites.py if you haven't.", file=sys.stderr)

# LiteLLM doesn't require a global configure call.
# API keys are typically set as environment variables.
# LiteLLM will automatically use the appropriate credentials based on the model ID prefix.

class LiteLLMMcpClient:
    """A client that connects to an MCP server and uses LiteLLM for interaction."""

    def __init__(self, system_instruction: Optional[str] = None):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Revert to LITELLM_MODEL_ID from os.getenv, as app_config is removed for this context
        self.model_name = os.getenv("LITELLM_MODEL_ID", "gemini/gemini-1.5-pro-latest")
        
        # Determine model provider based on model name prefix
        self.model_provider = "Unknown"
        if self.model_name.startswith("gemini/"):
            self.model_provider = "Google AI (Gemini)"
        elif self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
            self.model_provider = "OpenAI"
        elif self.model_name.startswith("bedrock/"):
            self.model_provider = "AWS Bedrock"
            # Extract the actual model name from bedrock format
            if "claude" in self.model_name.lower():
                self.model_provider += " (Anthropic Claude)"
        elif self.model_name.startswith("anthropic/") or self.model_name.startswith("claude"):
            self.model_provider = "Anthropic"
        elif self.model_name.startswith("ollama/"):
            self.model_provider = "Ollama (Local)"
        
        self.system_instruction_content = system_instruction
        self.conversation_history: list[Dict[str, Any]] = []
        if self.system_instruction_content:
            self.conversation_history.append({"role": "system", "content": self.system_instruction_content})

        # RAG and Display Thresholds are now hardcoded in vector_store_module.
        # Removing client-level attributes for these.
        # self.rag_similarity_threshold = vector_store_config.LITELLM_RAG_THRESHOLD (Removed)
        # self.display_similarity_threshold = vector_store_config.LITELLM_DISPLAY_THRESHOLD (Removed)
        
        # Memory directories are ensured by memory_module on its import.
        # No need for verbose printing of these paths here.

        # Core state variables
        self.is_initialized: bool = False
        self.current_db_connection_string: Optional[str] = None
        self.current_db_name_identifier: Optional[str] = None # e.g., "california_schools_db"
        
        # Data used for context
        self.db_schema_and_sample_data: Optional[Dict[str, Any]] = None # Loaded from memory/schema/
        self.cumulative_insights_content: Optional[str] = None # Loaded from memory/insights/summarized_insights.md

        # State for the current query/feedback cycle
        self.current_natural_language_question: Optional[str] = None
        # Holds the Pydantic model of the full feedback report being built/iterated on
        self.current_feedback_report_content: Optional[FeedbackReportContentModel] = None
        # Note: last_generated_sql, last_sql_explanation etc. are now part of current_feedback_report_content
        
        # State for query revision feature
        self.current_revision_report_content: Optional[RevisionReportContentModel] = None
        self.is_in_revision_mode: bool = False
        self.feedback_used_in_current_revision_cycle: bool = False
        self.feedback_log_in_revision: list[Dict[str, str]] = []
    
    def _get_first_run_flag_path(self):
        return os.path.join(memory_module.BASE_MEMORY_DIR, ".first_run_complete")

    def _check_and_set_first_run(self) -> bool:
        """Checks if this is the first run. If so, creates a flag file and returns True."""
        flag_path = self._get_first_run_flag_path()
        if not os.path.exists(flag_path):
            try:
                with open(flag_path, "w") as f:
                    f.write(datetime.datetime.now().isoformat())
                return True # It was the first run
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Could not create first run flag at {flag_path}: {e}{Style.RESET_ALL}", file=sys.stderr)
                return False # Assume not first run if cannot create flag
        return False # Flag exists, not first run

    async def _cleanup_database_session(self, full_cleanup: bool = True):
        # print("Cleaning up database session state...") # User doesn't need to see this
        if full_cleanup:
            self.is_initialized = False
            self.current_db_connection_string = None
            self.current_db_name_identifier = None
            self.db_schema_and_sample_data = None
            self.cumulative_insights_content = None 
        
        self._reset_feedback_cycle_state()
        self._reset_revision_cycle_state() # Added
        # print("Database session state cleaned.")

    def _reset_feedback_cycle_state(self):
        """Resets state for a new natural language query and its feedback cycle."""
        # print("Resetting feedback cycle state for new SQL generation...") # Internal detail
        self.current_natural_language_question = None
        self.current_feedback_report_content = None
        # If starting a new feedback cycle, revision mode should also reset
        self._reset_revision_cycle_state()


    def _reset_revision_cycle_state(self):
        """Resets state for a new query revision cycle."""
        # print("Resetting revision cycle state...") # Internal detail
        self.current_revision_report_content = None
        self.is_in_revision_mode = False
        self.feedback_used_in_current_revision_cycle = False
        self.feedback_log_in_revision = []

    def _extract_connection_string_and_db_name(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        # Extracts connection string and attempts to derive a db_name from it.
        # Example: postgresql://user:password@host:port/dbname
        conn_str_match = re.search(r"(postgresql://\S+:\S+@\S+:\d+/(\S+))", query)
        if conn_str_match:
            full_conn_str = conn_str_match.group(1)
            db_name_part = conn_str_match.group(2)
            # Sanitize db_name_part to be a valid filename component
            sanitized_db_name = re.sub(r'[^\w\-_\.]', '_', db_name_part) if db_name_part else "unknown_db"
            return full_conn_str, sanitized_db_name
        return None, None

    def _extract_mcp_tool_call_output(self, tool_call_result: Any) -> Any:
        if not self.session: return "Error: MCP session not available."
        if hasattr(tool_call_result, 'content') and isinstance(tool_call_result.content, list) and \
           len(tool_call_result.content) > 0 and hasattr(tool_call_result.content[0], 'text') and \
           tool_call_result.content[0].text is not None:
            return tool_call_result.content[0].text
        elif hasattr(tool_call_result, 'output'): return tool_call_result.output
        elif hasattr(tool_call_result, 'result'): return tool_call_result.result
        if tool_call_result is not None:
            if not isinstance(tool_call_result, (str, int, float, bool, list, dict)):
                try: return str(tool_call_result)
                except Exception as e: return f"Error: Tool output format not recognized and failed to convert to string. Type: {type(tool_call_result)}, Error: {e}"
            else: return tool_call_result
        return "Error: Tool output not found or format not recognized."

    async def connect_to_mcp_server(self, server_script_path: str):
        # print(f"Connecting to MCP server script: {server_script_path}") # Internal detail
        is_python = server_script_path.endswith('.py')
        command = sys.executable if is_python else "node"
        # If server_script_path is relative, make it relative to this script's directory
        if not os.path.isabs(server_script_path):
            server_script_path = os.path.join(os.path.dirname(__file__), server_script_path)

        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.session.initialize()
            response = await self.session.list_tools()
            mcp_tools_list: list[McpTool] = response.tools
            # print(f"Connected to MCP server with {len(mcp_tools_list)} tools: {[tool.name for tool in mcp_tools_list]}") # Internal detail
            if not mcp_tools_list: 
                print(f"{Fore.RED}Error: No tools discovered from the MCP server. Exiting.{Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)

            self.litellm_tools = []
            for mcp_tool_obj in mcp_tools_list:
                tool_name = getattr(mcp_tool_obj, 'name', None)
                tool_desc = getattr(mcp_tool_obj, 'description', None)
                tool_schema = copy.deepcopy(getattr(mcp_tool_obj, 'inputSchema', {}))
                
                if tool_schema:
                    if 'properties' not in tool_schema and tool_schema:
                        # print(f"Warning: Tool '{tool_name}' schema might not be directly OpenAI compatible: {tool_schema}") # Debug
                        pass
                    elif 'properties' in tool_schema:
                         tool_schema['type'] = 'object'
                    tool_schema.pop('title', None)
                    if 'properties' in tool_schema and isinstance(tool_schema['properties'], dict):
                        for prop_name in list(tool_schema['properties'].keys()):
                            if isinstance(tool_schema['properties'][prop_name], dict):
                                tool_schema['properties'][prop_name].pop('title', None)
                
                if tool_name and tool_desc:
                    self.litellm_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_desc,
                            "parameters": tool_schema or {"type": "object", "properties": {}}
                        }
                    })
                else:
                    print(f"{Fore.YELLOW}Warning: Skipping MCP tool '{tool_name}' due to missing attributes for LiteLLM conversion.{Style.RESET_ALL}", file=sys.stderr) # Keep this warning
            
            if not self.litellm_tools and mcp_tools_list:
                print(f"{Fore.RED}Error: No MCP tools could be converted to LiteLLM format. Exiting.{Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)
            # print(f"LiteLLM client configured to use model '{self.model_name}' with {len(self.litellm_tools)} tools.") # Internal detail
        except Exception as e:
            print(f"{Fore.RED}Fatal error during MCP server connection or LiteLLM setup: {e}. Please ensure the MCP server script is correct and executable.{Style.RESET_ALL}", file=sys.stderr)
            await self.cleanup()
            sys.exit(1)

    async def _send_message_to_llm(self, messages: list, tools: Optional[list] = None, tool_choice: str = "auto", 
                                  response_format: Optional[dict] = None) -> Any:
        """Sends messages to LiteLLM and handles response, including tool calls."""
        try:
            # print(f"Sending to LiteLLM ({self.model_name}). Messages: {json.dumps(messages, indent=2)[:200]}...") # Debug, too verbose
            # if tools: # Debug
            #     print(f"Tools provided: {[tool['function']['name'] for tool in tools]}")
            # print(f"{Fore.MAGENTA}DEBUG: Sending to LLM. Model: {self.model_name}{Style.RESET_ALL}") # Optional debug
            
            # Prepare kwargs for acompletion
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            
            # Add response_format if provided (for JSON responses with OpenAI models)
            if response_format and (self.model_name.startswith("gpt-") or self.model_name.startswith("openai/")):
                kwargs["response_format"] = response_format
            
            response = await acompletion(**kwargs)
            
            # Add user message to history (assistant response added after processing)
            # The last message in `messages` is the current user prompt
            if messages[-1]["role"] == "user":
                 self.conversation_history.append(messages[-1])

            return response
        except Exception as e:
            # Log error to stderr instead of displaying to user
            print(f"{Fore.RED}Error calling LiteLLM: {e}{Style.RESET_ALL}", file=sys.stderr)
            # Add user message to history even if call fails, to keep track
            if messages[-1]["role"] == "user" and messages[-1] not in self.conversation_history:
                 self.conversation_history.append(messages[-1])
            # Add a placeholder error message to history for the assistant
            self.conversation_history.append({"role": "assistant", "content": "I'm having trouble processing your request. Let me try again."})
            raise # Re-raise the exception to be handled by the caller

    async def _process_llm_response(self, llm_response: Any) -> Tuple[str, bool]:
        """Processes LiteLLM response, handles tool calls, and returns text response and if a tool was called."""
        assistant_response_content = ""
        tool_calls_made = False

        if not llm_response or not llm_response.choices or not llm_response.choices[0].message:
            assistant_response_content = f"{Fore.RED}Error: Empty or invalid response from LLM.{Style.RESET_ALL}"
            self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
            return assistant_response_content, tool_calls_made

        message = llm_response.choices[0].message
        
        # Storing the raw assistant message (potentially with tool_calls)
        assistant_message_for_history = {"role": "assistant"}
        if message.content:
            assistant_message_for_history["content"] = message.content
            assistant_response_content = message.content # Initial text part

        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls_made = True
            assistant_message_for_history["tool_calls"] = [] # For history
            
            # LiteLLM returns tool_calls in OpenAI format
            # [{"id": "call_abc", "type": "function", "function": {"name": "tool_name", "arguments": "{...}"}}]
            
            tool_call_responses_for_next_llm_call = []

            for tool_call in message.tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                
                # Add the tool call itself to history
                assistant_message_for_history["tool_calls"].append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args_str}
                })

                # print(f"LLM wants to call tool: {tool_name} with args: {tool_args_str}") # Debug
                try:
                    tool_args = json.loads(tool_args_str)
                    mcp_tool_result_obj = await self.session.call_tool(tool_name, tool_args)
                    tool_output = self._extract_mcp_tool_call_output(mcp_tool_result_obj)
                    # print(f"Tool {tool_name} output: {str(tool_output)[:200]}...") # Debug
                except json.JSONDecodeError as e_json:
                    tool_output = f"Error: Invalid JSON arguments for tool {tool_name}: {e_json}. Arguments received: {tool_args_str}"
                    # print(tool_output, file=sys.stderr) # Keep for critical errors
                except Exception as e_tool:
                    tool_output = f"Error executing tool {tool_name}: {e_tool}"
                    # print(tool_output, file=sys.stderr) # Keep for critical errors

                tool_call_responses_for_next_llm_call.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": str(tool_output) # Ensure content is string
                })

            # Add the assistant's message (that included the tool call request) to history
            self.conversation_history.append(assistant_message_for_history)

            # Add all tool responses to history
            for resp in tool_call_responses_for_next_llm_call:
                self.conversation_history.append(resp) 

            # Make a follow-up call to LLM with tool responses
            # print("Sending tool responses back to LLM...") # Debug
            
            # The conversation history already contains:
            # ..., user_prompt, assistant_tool_call_request, tool_response_1, ...
            # So we can just send the current self.conversation_history
            
            follow_up_llm_response = await self._send_message_to_llm(self.conversation_history, tools=self.litellm_tools) # Pass history
            
            # Process this new response (which should be the final text from LLM after tools)
            # This recursive call is safe as long as LLM doesn't loop infinitely on tool calls
            assistant_response_content, _ = await self._process_llm_response(follow_up_llm_response)
            # The _process_llm_response will handle adding the final assistant message to history.

        else: # No tool calls, just a direct text response
            if message.content: # Ensure there's content
                assistant_response_content = message.content
                # Add assistant's direct response to history
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
            else: # Should not happen if the first check passed, but as a safeguard
                assistant_response_content = "Error: LLM response message has no content."
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})


        return assistant_response_content, tool_calls_made


    async def _handle_initialization(self, connection_string: str, db_name_id: str) -> str:
        """Handles the full DB initialization flow."""
        await self._cleanup_database_session(full_cleanup=True) # Clean slate for new DB
        
        success, message, schema_data = await initialization_module.perform_initialization(
            self, connection_string, db_name_id
        )
        if success:
            self.is_initialized = True
            self.current_db_connection_string = connection_string
            self.current_db_name_identifier = db_name_id
            self.db_schema_and_sample_data = schema_data
            # Load cumulative insights for this specific DB if they exist
            self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
            self._reset_feedback_cycle_state() # Ensure clean state for new DB
            self._reset_revision_cycle_state() # Ensure clean state for new DB
            # print(f"Initialization successful for {db_name_id}. Insights for this DB loaded (if any).") # User sees the main message
            return message
        else:
            await self._cleanup_database_session(full_cleanup=True) 
            return message

    async def dispatch_command(self, query: str) -> str:
        if not self.session: # LiteLLM client doesn't use self.chat in the same way
            return "Error: Client not fully initialized (MCP session missing)."

        # Command: /change_database: "connection_string" or just "connection_string" for implicit init
        conn_str_explicit_match = re.match(r"/change_database:\s*(.+)", query, re.IGNORECASE) # Allow unquoted connection string
        parsed_conn_str, parsed_db_name_id = None, None

        if conn_str_explicit_match:
            raw_conn_str = conn_str_explicit_match.group(1).strip() 
            if raw_conn_str.startswith('"') and raw_conn_str.endswith('"'):
                raw_conn_str = raw_conn_str[1:-1]
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(raw_conn_str)
            if not parsed_conn_str: return "Error: Invalid connection string format provided with /change_database."
            # print(f"Received /change_database for {parsed_db_name_id}...") # User sees the result
            return await self._handle_initialization(parsed_conn_str, parsed_db_name_id)
        
        if not self.is_initialized:
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(query)
            if parsed_conn_str:
                # print(f"Attempting implicit initialization for {parsed_db_name_id}...") # User sees the result
                return await self._handle_initialization(parsed_conn_str, parsed_db_name_id)
            else:
                return "Error: Database not initialized. Please provide a connection string (e.g., postgresql://user:pass@host:port/dbname) or use /change_database."

        if not self.is_initialized or not self.current_db_name_identifier or self.db_schema_and_sample_data is None:
             # This state should ideally not be reached if logic above is correct
            return "Critical Error: Database initialization state is inconsistent. Please try /change_database again."

        # Command: /generate_sql: {Natural language question}
        generate_sql_match = re.match(r"/generate_sql:\s*(.+)", query, re.IGNORECASE)
        if generate_sql_match:
            nl_question = generate_sql_match.group(1).strip()
            # Reset both feedback and revision states for a new /generate_sql command
            self._reset_feedback_cycle_state()
            self._reset_revision_cycle_state()
            self.current_natural_language_question = nl_question
            # print(f"Received /generate_sql for: {nl_question}") # User sees the result

            print("Generating SQL, please wait...") # User-facing wait message
            sql_gen_result_dict = await sql_generation_module.generate_sql_query(
                self, nl_question, self.db_schema_and_sample_data, self.cumulative_insights_content
            )
            
            # Populate FeedbackReportContentModel (initial state)
            if sql_gen_result_dict.get("sql_query"):
                self.current_feedback_report_content = FeedbackReportContentModel(
                    natural_language_question=nl_question,
                    initial_sql_query=sql_gen_result_dict["sql_query"],
                    initial_explanation=sql_gen_result_dict.get("explanation", "N/A"),
                    final_corrected_sql_query=sql_gen_result_dict["sql_query"], # Initially same
                    final_explanation=sql_gen_result_dict.get("explanation", "N/A") # Initially same
                    # Other fields like analysis will be filled by LLM during feedback or approval
                )
                # TODO: Consider an LLM call here to fill `why_initial_query_was_wrong_or_suboptimal` etc.
                # if the initial query had an execution error. For now, these are blank.
            else: # SQL generation failed at the first step
                 self.current_feedback_report_content = None # Ensure it's cleared
            
            base_message_to_user = sql_gen_result_dict.get("message_to_user", "Error: No message from SQL generation.")
            
            # --- Augment message with display few-shot examples ---
            if self.current_db_name_identifier and self.current_natural_language_question:
                try:
                    # Using hardcoded display threshold from vector_store_module
                    display_threshold_from_module = vector_store_module.LITELLM_DISPLAY_THRESHOLD
                    display_examples = vector_store_module.search_similar_nlqs(
                        db_name_identifier=self.current_db_name_identifier,
                        query_nlq=self.current_natural_language_question,
                        k=3, # Show up to 3 examples
                        threshold=display_threshold_from_module
                    )
                    if display_examples:
                        display_message_parts = ["\n\n--- Relevant Approved Examples (Similarity >= " f"{display_threshold_from_module}" ") ---"]
                        for i, ex in enumerate(display_examples):
                            display_message_parts.append(f"Example {i+1} (Similarity: {ex['similarity_score']:.2f}):")
                            display_message_parts.append(f"  Q: \"{ex['nlq']}\"")
                            display_message_parts.append(f"  A: ```sql\n{ex['sql']}\n```")
                        base_message_to_user += "\n" + "\n".join(display_message_parts)
                except Exception as e_display_rag:
                    print(f"Error retrieving/formatting display RAG examples: {e_display_rag}", file=sys.stderr)
            # --- End Augment ---
            return base_message_to_user

        # Command: /feedback: {user_feedback_text}
        feedback_match = re.match(r"/feedback:\s*(.+)", query, re.IGNORECASE)
        if feedback_match:
            user_feedback_text = feedback_match.group(1).strip()
            print("Processing feedback, please wait...")

            if self.is_in_revision_mode and self.current_revision_report_content and self.current_revision_report_content.final_revised_sql_query:
                # Apply feedback to the current revision
                current_sql = self.current_revision_report_content.final_revised_sql_query
                current_explanation = self.current_revision_report_content.final_revised_explanation or "N/A"
                
                # Prompt for correcting SQL based on feedback (within revision context)
                feedback_prompt_for_revision = (
                    f"You are an expert PostgreSQL SQL assistant. A user is providing feedback on a previously revised SQL query.\n"
                    f"CURRENT REVISED SQL QUERY:\n```sql\n{current_sql}\n```\n"
                    f"ITS EXPLANATION:\n{current_explanation}\n\n"
                    f"USER FEEDBACK: \"{user_feedback_text}\"\n\n"
                    f"Based on this feedback, please provide a corrected SQL query (must start with SELECT) and a brief explanation for the correction.\n"
                    f"Respond ONLY with a single JSON object matching this structure: "
                    f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation for the correction>\" }}\n"
                )
                MAX_FEEDBACK_RETRIES_IN_REVISION = 1
                corrected_sql_from_feedback = None
                corrected_explanation_from_feedback = None

                for attempt in range(MAX_FEEDBACK_RETRIES_IN_REVISION + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt_for_revision}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        llm_response_obj = await self._send_message_to_llm(messages=messages_for_llm, response_format=response_format)
                        response_text, _ = await self._process_llm_response(llm_response_obj)

                        if response_text.startswith("```json"): response_text = response_text[7:]
                        if response_text.endswith("```"): response_text = response_text[:-3]
                        
                        parsed_correction = SQLGenerationResponse.model_validate_json(response_text.strip())
                        
                        if not parsed_correction.sql_query or not parsed_correction.sql_query.strip().upper().startswith("SELECT"):
                            raise ValueError("Corrected SQL from feedback must start with SELECT.")
                        
                        corrected_sql_from_feedback = parsed_correction.sql_query
                        corrected_sql_from_feedback = parsed_correction.sql_query
                        corrected_explanation_from_feedback = parsed_correction.explanation or "N/A"

                        # Log feedback for potential report generation on approve_revision
                        self.feedback_used_in_current_revision_cycle = True
                        self.feedback_log_in_revision.append({
                            "user_feedback_text": user_feedback_text,
                            "sql_before_feedback": current_sql, # SQL before this feedback
                            "explanation_before_feedback": current_explanation, # Explanation for current_sql
                            "corrected_sql_attempt": corrected_sql_from_feedback, # SQL after this feedback
                            "corrected_explanation": corrected_explanation_from_feedback # Explanation for new SQL
                        })
                        break 
                    except (ValidationError, json.JSONDecodeError, ValueError) as e:
                        if attempt == MAX_FEEDBACK_RETRIES_IN_REVISION:
                            return f"Error processing feedback on revised query: {e}. Please try rephrasing your feedback."
                        feedback_prompt_for_revision = f"Your previous attempt to correct the SQL based on feedback was invalid (Error: {e}). Please try again, ensuring the JSON output has 'sql_query' (starting with SELECT) and 'explanation'."
                    except Exception as e_gen:
                        if attempt == MAX_FEEDBACK_RETRIES_IN_REVISION:
                            return f"Unexpected error processing feedback on revised query: {e_gen}."
                        feedback_prompt_for_revision = "An unexpected error occurred. Please try to regenerate the corrected SQL and explanation based on the feedback."
                
                if corrected_sql_from_feedback and self.current_revision_report_content:
                    new_iteration = RevisionIteration(
                        user_revision_prompt=f"Feedback: {user_feedback_text}", 
                        revised_sql_attempt=corrected_sql_from_feedback,
                        revised_explanation=corrected_explanation_from_feedback
                    )
                    self.current_revision_report_content.revision_iterations.append(new_iteration)
                    self.current_revision_report_content.final_revised_sql_query = corrected_sql_from_feedback
                    self.current_revision_report_content.final_revised_explanation = corrected_explanation_from_feedback

                    exec_result, exec_error = None, None
                    try:
                        exec_obj = await self.session.call_tool("execute_postgres_query", {"query": corrected_sql_from_feedback})
                        raw_output = self._extract_mcp_tool_call_output(exec_obj)
                        if isinstance(raw_output, str) and "Error:" in raw_output: exec_error = raw_output
                        else: exec_result = raw_output
                    except Exception as e_exec: exec_error = str(e_exec)

                    user_msg = f"Feedback applied to revised query. New SQL attempt:\n```sql\n{corrected_sql_from_feedback}\n```\n"
                    user_msg += f"Explanation:\n{corrected_explanation_from_feedback}\n"
                    if exec_error: user_msg += f"\nExecution Error for new SQL: {exec_error}\n"
                    else: user_msg += f"\nExecution of new SQL successful. Result preview: {str(exec_result)[:200]}...\n"
                    user_msg += "Use `/revise Your new prompt`, more `/feedback`, or `/approve_revision`."
                    return user_msg
                else:
                    return "Failed to apply feedback to the revised query."

            elif self.current_feedback_report_content and self.current_natural_language_question:
                self._reset_revision_cycle_state() 
                current_report_json = self.current_feedback_report_content.model_dump_json(indent=2)
                feedback_model_schema_dict = FeedbackReportContentModel.model_json_schema()
                feedback_model_schema_str = json.dumps(feedback_model_schema_dict, indent=2)

                feedback_prompt = (
                    f"You are refining a SQL query based on user feedback and updating a detailed report.\n"
                    f"The user's original question was: \"{self.current_natural_language_question}\"\n"
                    f"The current state of the feedback report (JSON format) is:\n```json\n{current_report_json}\n```\n"
                    f"The user has provided new feedback: \"{user_feedback_text}\"\n\n"
                    f"Your tasks:\n"
                    f"1. Generate a new `corrected_sql_attempt` and `corrected_explanation` based on this latest feedback and the *previous* `final_corrected_sql_query` from the report.\n"
                    f"2. Create a new `FeedbackIteration` object containing this `user_feedback_text`, your new `corrected_sql_attempt`, and `corrected_explanation`.\n"
                    f"3. Append this new `FeedbackIteration` to the `feedback_iterations` list in the report.\n"
                    f"4. Update the report's `final_corrected_sql_query` and `final_explanation` to your latest attempt.\n"
                    f"5. Re-evaluate and update the LLM analysis sections: `why_initial_query_was_wrong_or_suboptimal` (if applicable, comparing to initial), `why_final_query_works_or_is_improved` (explaining your latest correction), `database_insights_learned_from_this_query`, and `sql_lessons_learned_from_this_query` based on the *entire* history including this new iteration.\n\n"
                    f"Respond ONLY with the complete, updated JSON object for the `FeedbackReportContentModel`, conforming to this schema:\n"
                    f"```json\n{feedback_model_schema_str}\n```\n"
                    f"Ensure the `corrected_sql_attempt` and `final_corrected_sql_query` start with SELECT."
                )
                
                MAX_FEEDBACK_RETRIES = 1
                for attempt in range(MAX_FEEDBACK_RETRIES + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        llm_response_obj = await self._send_message_to_llm(
                            messages=messages_for_llm,
                            response_format=response_format
                        )
                        response_text, _ = await self._process_llm_response(llm_response_obj)

                        if response_text.startswith("```json"): response_text = response_text[7:]
                        if response_text.endswith("```"): response_text = response_text[:-3]
                        
                        updated_report_model = FeedbackReportContentModel.model_validate_json(response_text.strip())
                        
                        if not updated_report_model.final_corrected_sql_query or \
                           not updated_report_model.final_corrected_sql_query.strip().upper().startswith("SELECT"):
                            raise ValueError("Corrected SQL in feedback report must start with SELECT.")

                        self.current_feedback_report_content = updated_report_model
                        
                        exec_result, exec_error = None, None
                        try:
                            exec_obj = await self.session.call_tool("execute_postgres_query", {"query": updated_report_model.final_corrected_sql_query})
                            raw_output = self._extract_mcp_tool_call_output(exec_obj)
                            if isinstance(raw_output, str) and "Error:" in raw_output: exec_error = raw_output
                            else: exec_result = raw_output
                        except Exception as e_exec: exec_error = str(e_exec)

                        user_msg = f"Feedback processed. New SQL attempt:\n```sql\n{updated_report_model.final_corrected_sql_query}\n```\n"
                        user_msg += f"Explanation:\n{updated_report_model.final_explanation}\n"
                        if exec_error: user_msg += f"\nExecution Error for new SQL: {exec_error}\n"
                        else: user_msg += f"\nExecution of new SQL successful. Result preview: {str(exec_result)[:200]}...\n"
                        user_msg += "Provide more /feedback or use /approved to save."
                        return user_msg

                    except (ValidationError, json.JSONDecodeError, ValueError) as e:
                        if attempt == MAX_FEEDBACK_RETRIES:
                            return f"I'm having trouble processing your feedback. Error: {e}. Could you provide your feedback again, perhaps with more specific details?"
                        feedback_prompt = f"Your previous response for updating the feedback report was invalid. Error: {e}. Please try again."
                    except Exception as e_gen:
                        if attempt == MAX_FEEDBACK_RETRIES:
                            return f"I encountered an issue while processing your feedback: {e_gen}. Could you rephrase your feedback?"
                        feedback_prompt = "An unexpected error occurred. Please try to regenerate the updated feedback report JSON."
            else:
                return "Error: No SQL query generated yet for feedback. Use /generate_sql first."


        # Command: /approved
        if query.strip().lower() == "/approved":
            if not self.current_feedback_report_content:
                return "Error: No feedback report to approve. Use /generate_sql first."

            print("Processing approval and updating memory, please wait...") # User-facing wait message
            try:
                # 1. Save Feedback Markdown
                feedback_filepath = memory_module.save_feedback_markdown(
                    self.current_feedback_report_content, 
                    self.current_db_name_identifier
                )
                # print(f"Feedback report saved to: {feedback_filepath}") # User sees final message

                saved_feedback_md_content = memory_module.read_feedback_file(feedback_filepath)
                if not saved_feedback_md_content:
                    return "Error: Could not read back saved feedback file for insights processing."

                insights_success = await insights_module.generate_and_update_insights(
                    self, 
                    saved_feedback_md_content,
                    self.current_db_name_identifier
                )
                if insights_success:
                    # print("Insights successfully generated/updated.") # User sees final message
                    self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
                else:
                    print("Warning: Failed to generate or update insights from this feedback.", file=sys.stderr) # Keep but redirect to stderr
                
                if self.current_natural_language_question and self.current_feedback_report_content.final_corrected_sql_query:
                    try:
                        memory_module.save_nl2sql_pair(
                            self.current_db_name_identifier,
                            self.current_natural_language_question,
                            self.current_feedback_report_content.final_corrected_sql_query
                        )
                        # print("NLQ-SQL pair saved.") # User sees final message
                    except Exception as e_nl2sql:
                        print(f"Warning: Failed to save NLQ-SQL pair: {e_nl2sql}", file=sys.stderr) # Keep but redirect to stderr
                else:
                    print("Warning: Could not save NLQ-SQL pair due to missing NLQ or final SQL query in the report.", file=sys.stderr) # Keep but redirect to stderr

                final_user_message = f"Approved. Feedback report, insights, and NLQ-SQL pair for '{self.current_db_name_identifier}' saved."
                # Remove technical error message about insights update failure
                
                self._reset_feedback_cycle_state()
                self._reset_revision_cycle_state() # Also reset revision state on approval
                return f"{final_user_message}\nYou can start a new query with /generate_sql."

            except Exception as e:
                return f"Error during approval process: {e}"

        # Command: /revise: {revision_prompt}
        revise_match = re.match(r"/revise:\s*(.+)", query, re.IGNORECASE)
        if revise_match:
            revision_prompt = revise_match.group(1).strip()
            
            sql_to_start_revision_with = None
            if self.is_in_revision_mode and self.current_revision_report_content and self.current_revision_report_content.final_revised_sql_query:
                sql_to_start_revision_with = self.current_revision_report_content.final_revised_sql_query
            elif self.current_feedback_report_content and self.current_feedback_report_content.final_corrected_sql_query:
                sql_to_start_revision_with = self.current_feedback_report_content.final_corrected_sql_query
            
            if not sql_to_start_revision_with:
                return "Error: No SQL query available to revise. Use /generate_sql first, or ensure a previous revision/feedback cycle completed with a query."

            if not self.is_in_revision_mode:
                self.is_in_revision_mode = True
                # self.current_feedback_report_content = None # Decided against this to allow revising feedback results
                self.current_revision_report_content = RevisionReportContentModel(
                    initial_sql_for_revision=sql_to_start_revision_with
                )
            
            print(f"Revising SQL based on: \"{revision_prompt}\", please wait...")

            revision_history_for_llm_context = []
            if self.current_revision_report_content:
                for rev_iter in self.current_revision_report_content.revision_iterations:
                    revision_history_for_llm_context.append({"role": "user", "content": rev_iter.user_revision_prompt})
                    revision_history_for_llm_context.append({"role": "assistant", "content": f"Revised SQL: {rev_iter.revised_sql_attempt}\nExplanation: {rev_iter.revised_explanation}"})
            
            current_sql_for_this_iteration = self.current_revision_report_content.final_revised_sql_query or self.current_revision_report_content.initial_sql_for_revision
            
            revision_result = await revise_query_module.handle_revise_query_iteration(
                self,
                user_revision_prompt=revision_prompt,
                current_sql_to_revise=current_sql_for_this_iteration,
                revision_history_for_context=revision_history_for_llm_context
            )

            if revision_result.get("revised_sql_query") and self.current_revision_report_content:
                new_iteration = RevisionIteration(
                    user_revision_prompt=revision_prompt,
                    revised_sql_attempt=revision_result["revised_sql_query"],
                    revised_explanation=revision_result.get("revised_explanation", "N/A")
                )
                self.current_revision_report_content.revision_iterations.append(new_iteration)
                self.current_revision_report_content.final_revised_sql_query = revision_result["revised_sql_query"]
                self.current_revision_report_content.final_revised_explanation = revision_result.get("revised_explanation", "N/A")
            
            return revision_result.get("message_to_user", "Error in revision process.")

        # Command: /approve_revision
        if query.strip().lower() == "/approve_revision":
            if not self.is_in_revision_mode or not self.current_revision_report_content or not self.current_revision_report_content.final_revised_sql_query:
                return "Error: No active revision cycle or final revised SQL to approve. Use /revise first."

            print("Finalizing revision and generating NLQ, please wait...")
            final_sql = self.current_revision_report_content.final_revised_sql_query
            
            nlq_gen_result = await revise_query_module.generate_nlq_for_revised_sql(
                self,
                final_revised_sql=final_sql,
                revision_report=self.current_revision_report_content
            )

            if nlq_gen_result.get("generated_nlq") and self.current_db_name_identifier and final_sql and self.current_revision_report_content:
                generated_nlq = nlq_gen_result["generated_nlq"]
                self.current_revision_report_content.llm_generated_nlq_for_final_sql = generated_nlq
                self.current_revision_report_content.llm_reasoning_for_nlq = nlq_gen_result.get("reasoning")
                
                user_msg_parts = [
                    f"Revision approved. Final SQL:\n```sql\n{final_sql}\n```",
                    f"LLM-generated Natural Language Question for this SQL:\n\"{generated_nlq}\"",
                    f"(Reasoning: {nlq_gen_result.get('reasoning', 'N/A')})"
                ]

                try:
                    memory_module.save_nl2sql_pair(
                        self.current_db_name_identifier,
                        generated_nlq,
                        final_sql
                    )
                    user_msg_parts.append(f"This NLQ-SQL pair has been saved for '{self.current_db_name_identifier}'.")

                    # --- New: Generate feedback report and insights if feedback was used in revision ---
                    if self.feedback_used_in_current_revision_cycle and self.feedback_log_in_revision:
                        print("Processing feedback from revision cycle for report and insights...") # User-facing message
                        
                        report_feedback_iterations = [
                            FeedbackIteration(
                                user_feedback_text=log_item["user_feedback_text"],
                                corrected_sql_attempt=log_item["corrected_sql_attempt"],
                                corrected_explanation=log_item["corrected_explanation"]
                            ) for log_item in self.feedback_log_in_revision
                        ]

                        # Construct a temporary FeedbackReportContentModel for analysis
                        temp_feedback_report_for_revision_analysis = FeedbackReportContentModel(
                            natural_language_question=generated_nlq,
                            initial_sql_query=self.feedback_log_in_revision[0]["sql_before_feedback"],
                            initial_explanation=self.feedback_log_in_revision[0]["explanation_before_feedback"],
                            feedback_iterations=report_feedback_iterations,
                            final_corrected_sql_query=self.feedback_log_in_revision[-1]["corrected_sql_attempt"], # Should be same as final_sql
                            final_explanation=self.feedback_log_in_revision[-1]["corrected_explanation"],
                        )
                        
                        feedback_model_schema_dict_for_analysis = FeedbackReportContentModel.model_json_schema()
                        feedback_model_schema_str_for_analysis = json.dumps(feedback_model_schema_dict_for_analysis, indent=2)
                        current_report_json_for_analysis = temp_feedback_report_for_revision_analysis.model_dump_json(indent=2)

                        analysis_prompt = (
                            f"You are an AI assistant. A user has gone through a query revision process, which included providing feedback that led to corrections. "
                            f"The final revised SQL has been approved, and its corresponding natural language question (NLQ) has been generated.\n"
                            f"NLQ for the final approved SQL: \"{generated_nlq}\"\n"
                            f"The state of the feedback report *before* your analysis (detailing feedback within the revision cycle) is:\n```json\n{current_report_json_for_analysis}\n```\n"
                            f"Your task is to populate the analytical fields of this feedback report: "
                            f"`why_initial_query_was_wrong_or_suboptimal` (comparing the 'initial_sql_query' in the report to the 'final_corrected_sql_query' based on the feedback), "
                            f"`why_final_query_works_or_is_improved` (explaining why the 'final_corrected_sql_query' is better due to the feedback), "
                            f"`database_insights_learned_from_this_query`, and `sql_lessons_learned_from_this_query`.\n\n"
                            f"Respond ONLY with the complete, updated JSON object for the `FeedbackReportContentModel`, conforming to this schema:\n"
                            f"```json\n{feedback_model_schema_str_for_analysis}\n```\n"
                            f"Ensure all fields from the input report JSON are preserved, and only the analysis fields are added/updated."
                        )
                        
                        MAX_ANALYSIS_RETRIES = 1
                        updated_feedback_report_from_revision: Optional[FeedbackReportContentModel] = None

                        for analysis_attempt in range(MAX_ANALYSIS_RETRIES + 1):
                            try:
                                analysis_messages = self.conversation_history + [{"role": "user", "content": analysis_prompt}]
                                response_format_analysis = None
                                if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                                    response_format_analysis = {"type": "json_object"}
                                
                                analysis_llm_response_obj = await self._send_message_to_llm(
                                    messages=analysis_messages, response_format=response_format_analysis
                                )
                                analysis_llm_response_text, _ = await self._process_llm_response(analysis_llm_response_obj)

                                if analysis_llm_response_text.startswith("```json"): analysis_llm_response_text = analysis_llm_response_text[7:]
                                if analysis_llm_response_text.endswith("```"): analysis_llm_response_text = analysis_llm_response_text[:-3]
                                
                                updated_feedback_report_from_revision = FeedbackReportContentModel.model_validate_json(analysis_llm_response_text.strip())
                                break 
                            except (ValidationError, json.JSONDecodeError, ValueError) as e_analysis:
                                if analysis_attempt == MAX_ANALYSIS_RETRIES:
                                    user_msg_parts.append(f"Warning: Failed to get LLM analysis for revision feedback report: {e_analysis}. Report will be saved without full analysis.")
                                    updated_feedback_report_from_revision = temp_feedback_report_for_revision_analysis
                                    break
                                analysis_prompt = f"Your previous attempt to provide analysis for the feedback report was invalid. Error: {e_analysis}. Please try again."
                            except Exception as e_gen_analysis:
                                if analysis_attempt == MAX_ANALYSIS_RETRIES:
                                    user_msg_parts.append(f"Warning: Unexpected error during LLM analysis for revision feedback report: {e_gen_analysis}. Report will be saved without full analysis.")
                                    updated_feedback_report_from_revision = temp_feedback_report_for_revision_analysis
                                    break
                                analysis_prompt = "An unexpected error occurred during analysis. Please try again."
                        
                        if updated_feedback_report_from_revision:
                            try:
                                feedback_filepath_rev = memory_module.save_feedback_markdown(
                                    updated_feedback_report_from_revision, self.current_db_name_identifier
                                )
                                user_msg_parts.append(f"Feedback report for revision cycle saved: {os.path.basename(feedback_filepath_rev)}")

                                saved_feedback_md_content_rev = memory_module.read_feedback_file(feedback_filepath_rev)
                                if saved_feedback_md_content_rev:
                                    insights_success_rev = await insights_module.generate_and_update_insights(
                                        self, saved_feedback_md_content_rev, self.current_db_name_identifier
                                    )
                                    if insights_success_rev:
                                        user_msg_parts.append("Insights updated from revision feedback.")
                                        self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
                                    else:
                                        user_msg_parts.append("Warning: Failed to update insights from revision feedback.")
                                else:
                                    user_msg_parts.append("Warning: Could not read back revision feedback file for insights processing.")
                            except Exception as e_fb_insights_rev:
                                user_msg_parts.append(f"Error during feedback report/insights saving for revision: {e_fb_insights_rev}")
                        else:
                             user_msg_parts.append("Warning: Could not generate the full feedback report for the revision cycle.")
                    # --- End new section ---

                    user_msg_parts.append("You can start a new query with /generate_sql.")
                    self._reset_revision_cycle_state() # Also resets the new feedback flags
                    return "\n".join(user_msg_parts)
                
                except Exception as e_save:
                    return f"Error saving approved revision NLQ-SQL pair: {e_save}"
            else:
                return nlq_gen_result.get("message_to_user", "Error: Could not generate NLQ for the revised SQL.")
        
        if query.strip().lower() == "/list_commands":
            return self._get_commands_help_text()

        if not query.startswith("/"):
            # DO NOT reset feedback/revision state here.
            # Allow navigation queries without breaking the current SQL refinement loop.
            return await database_navigation_module.handle_navigation_query(
                self, query, self.current_db_name_identifier, 
                self.cumulative_insights_content, self.db_schema_and_sample_data
            )

        return (f"Unknown command: '{query.split()[0]}'. Current database: {self.current_db_name_identifier or 'None'}.\n"
                f"Type /list_commands for available commands.")

    def _get_commands_help_text(self) -> str:
        """Returns the help text for commands."""
        return (
            "Available commands:\n"
            "  /change_database: \"postgresql://user:pass@host:port/dbname\"\n"
            "    - Connects to a new PostgreSQL database or changes the active one.\n"
            "  /generate_sql: {Your natural language question for SQL}\n"
            "    - Generates a SQL query based on your question.\n"
            "  /feedback: {Your feedback on the last generated SQL}\n"
            "    - Provide feedback to refine the last SQL query from /generate_sql or /feedback.\n"
            "  /approved\n"
            "    - Approves the last SQL query from /generate_sql or /feedback. Saves report, insights, and NLQ-SQL pair.\n"
            "  /revise: {Your revision prompt for the last SQL query}\n"
            "    - Iteratively revises the last generated/fed-back/revised SQL query.\n"
            "  /approve_revision\n"
            "    - Approves the final SQL from a /revise cycle. Generates an NLQ for it and saves the pair.\n"
            "  /list_commands\n"
            "    - Shows this list of available commands.\n"
            "  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights (resets feedback/revision state)."
        )

    async def chat_loop(self):
        print(f"\n{Fore.MAGENTA}PostgreSQL Co-Pilot Started!{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Using LiteLLM with model: {Style.BRIGHT}{self.model_name}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Model provider: {Style.BRIGHT}{self.model_provider}{Style.RESET_ALL}")
        
        is_first_run = self._check_and_set_first_run()
        if is_first_run:
            print(f"\n{Fore.YELLOW}Welcome! It looks like this is your first time running the Co-Pilot.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please ensure you have run 'python prerequisites.py' to set up necessary folders and get API key instructions.{Style.RESET_ALL}")
            print(self._get_commands_help_text()) # Show full help on first run
        else:
            # On subsequent runs, just remind them about /list_commands
            print(f"\n{Fore.CYAN}Type '/list_commands' to see available commands.{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Type a PostgreSQL connection string (e.g., postgresql://user:pass@host:port/dbname) to begin,{Style.RESET_ALL}")
        print(f"{Fore.CYAN}or use '/change_database: \"your_connection_string\"'. Type 'quit' to exit.{Style.RESET_ALL}")

        while True:
            try:
                prompt_text = f"\n{Fore.GREEN}[{self.current_db_name_identifier or f'{Style.DIM}No DB{Style.NORMAL}'}] {Style.BRIGHT}Query:{Style.RESET_ALL} "
                user_input_query = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(prompt_text).strip()
                )
                if user_input_query.lower() == 'quit':
                    await self._cleanup_database_session(full_cleanup=True)
                    break
                if not user_input_query: continue
                
                # Add user input to conversation history before dispatching
                # self.conversation_history.append({"role": "user", "content": user_input_query})
                # Dispatch command will handle adding to history via _send_message_to_llm

                response_text = await self.dispatch_command(user_input_query)
                print(f"\n{Fore.CYAN}{Style.BRIGHT}AI:{Style.RESET_ALL} {response_text}")

            except EOFError: print(f"\n{Fore.RED}Exiting chat loop.{Style.RESET_ALL}"); break
            except Exception as e: 
                print(f"\n{Fore.RED}Error in chat loop: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
                print(f"\n{Fore.YELLOW}I encountered an issue. Let's try again.{Style.RESET_ALL}")

    async def cleanup(self):
        print("\nCleaning up client resources...")
        await self._cleanup_database_session(full_cleanup=True)
        await self.exit_stack.aclose()
        print("Client cleanup complete.")

async def main():
    # Set LiteLLM verbosity (optional)
    # litellm.set_verbose = True # or litellm.success_callback = [...] etc.
    # litellm.telemetry = False # Disable LiteLLM telemetry if desired
    colorama_init(autoreset=True) # Initialize Colorama

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')

    if len(sys.argv) < 2:
        print("Usage: python postgres_copilot_chat.py <path_to_mcp_server_script.py>", file=sys.stderr)
        print("Example: python postgres_copilot_chat.py ./postgresql_server.py", file=sys.stderr)
        sys.exit(1)
    
    mcp_server_script_arg = sys.argv[1]
    # If mcp_server_script_arg is relative, it's relative to CWD.
    # If this script (postgres_copilot_chat.py) is in db-copilot, and server is also in db-copilot,
    # and user runs from parent dir: python db-copilot/postgres_copilot_chat.py db-copilot/postgresql_server.py
    # then sys.argv[1] will be "db-copilot/postgresql_server.py".
    # If user runs from db-copilot dir: python postgres_copilot_chat.py postgresql_server.py
    # then sys.argv[1] will be "postgresql_server.py".
    # The connect_to_mcp_server method now makes server_script_path absolute relative to *this* script's dir if it's relative.
    
    mcp_server_script = mcp_server_script_arg # Use the argument directly as connect_to_mcp_server will resolve it

    if not os.path.exists(mcp_server_script):
        # Try resolving relative to this script's directory if not found from CWD
        # This helps if the user provides a path relative to the script itself,
        # e.g. running `python main_script.py ../servers/mcp_server.py` from a `src` dir.
        # However, for `python db-copilot/pg_chat.py db-copilot/pg_server.py` from parent,
        # `mcp_server_script_arg` is already correct relative to CWD.
        # The logic in `connect_to_mcp_server` handles making it absolute if it's relative *to the script's dir*.
        # So, if it's not found as is, it might be an incorrect path.
        # Let's check if it's found relative to the script dir if not absolute.
        if not os.path.isabs(mcp_server_script_arg):
            path_relative_to_script = os.path.join(os.path.dirname(__file__), mcp_server_script_arg)
            if os.path.exists(path_relative_to_script):
                mcp_server_script = path_relative_to_script
            else:
                print(f"Error: MCP server script not found at '{mcp_server_script_arg}' (from CWD) or '{path_relative_to_script}' (relative to script).", file=sys.stderr)
                sys.exit(1)
        else: # Absolute path given but not found
            print(f"Error: MCP server script not found at absolute path '{mcp_server_script_arg}'", file=sys.stderr)
            sys.exit(1)


    custom_system_prompt = (
        "You are an expert PostgreSQL assistant. You will help users by generating SQL queries, "
        "analyzing feedback, and navigating database structures. Adhere strictly to JSON output formats when requested."
        "When you need to call a tool, you will be provided with a list of available tools. "
        "Your response should include a `tool_calls` field if you decide to use a tool. "
        "The `tool_calls` field should be a list of objects, each with `id`, `type: 'function'`, and `function` (containing `name` and `arguments` as a JSON string)."
    )
    # Specify the LiteLLM model name, e.g., "gpt-3.5-turbo", "gemini/gemini-pro", "claude-2"
    # This can also be set via an environment variable that LiteLLM reads, or passed as arg.
    # For now, using a Gemini model via LiteLLM as an example, assuming GOOGLE_API_KEY is set.
    # Or use "ollama/mistral" if ollama server is running with mistral.
    # Or "gpt-3.5-turbo" if OPENAI_API_KEY is set.
    # Defaulting to a common Gemini model that LiteLLM supports.
    # Model name is now read from .env inside LiteLLMMcpClient constructor
    client = LiteLLMMcpClient(system_instruction=custom_system_prompt)

    # The vector_store_module now uses its own hardcoded constants for thresholds.
    # No explicit call to configure_thresholds is needed here anymore.
    # Print a message indicating where these are now set.
    print(f"Vector store thresholds are now set directly in vector_store_module.py:")
    print(f"  RAG Threshold: {vector_store_module.LITELLM_RAG_THRESHOLD}")
    print(f"  Display Threshold: {vector_store_module.LITELLM_DISPLAY_THRESHOLD}")
    print(f"  Embedding Model: {vector_store_module.LITELLM_EMBEDDING_MODEL}")
    
    try:
        # memory_module.ensure_memory_directories() is called on its import.
        print(f"{Fore.BLUE}PostgreSQL Co-Pilot Client (LiteLLM) starting...{Style.RESET_ALL}")
        await client.connect_to_mcp_server(mcp_server_script) # Connect to the local postgresql_server.py
        
        # With LiteLLM, there's no explicit chat object to check like `client.chat`.
        # We assume connection is successful if connect_to_mcp_server doesn't exit.
        await client.chat_loop()

        # Save conversation history (now client.conversation_history)
        if client.conversation_history:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file_path = os.path.join(memory_module.CONVERSATION_HISTORY_DIR, f"conversation_litellm_{timestamp}.json")
            print(f"\n{Fore.CYAN}Saving conversation history to: {history_file_path}{Style.RESET_ALL}")
            try:
                with open(history_file_path, "w", encoding="utf-8") as f:
                    # Save as JSON for better structure, especially with tool calls
                    history_to_save = {
                        "db_name": client.current_db_name_identifier or "N/A",
                        "model_used": client.model_name,
                        "model_provider": client.model_provider,
                        "history": client.conversation_history
                    }
                    json.dump(history_to_save, f, indent=2)
                print(f"{Fore.GREEN}Conversation history saved.{Style.RESET_ALL}")
            except Exception as e_hist:
                print(f"{Fore.RED}Error saving conversation history: {e_hist}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No conversation history to save.{Style.RESET_ALL}")

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
