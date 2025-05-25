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
from pydantic_models import SQLGenerationResponse, FeedbackReportContentModel, FeedbackIteration #, InsightsExtractionModel

# --- Directory and .env Setup ---
# These setup steps are now primarily handled by prerequisites.py
# We still need to load .env here for the application to use the API key.
PASSWORDS_DIR = os.path.join(os.path.dirname(__file__), 'passwords')
dotenv_path = os.path.join(PASSWORDS_DIR, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # Minimal check, actual key functionality tested by LiteLLM
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print(f"Warning: No common API key (GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY) found in {dotenv_path}. Ensure the correct key for your LiteLLM model is set.")
else:
    print(f"Warning: .env file not found at {dotenv_path}. API keys for LiteLLM will likely be missing. Please run prerequisites.py if you haven't.", file=sys.stderr)

# LiteLLM doesn't require a global configure call.
# API keys are typically set as environment variables.
# Example: export OPENAI_API_KEY="your_key" or set in .env

# It's good practice to ensure at least one common key is present if we expect a specific provider.
# However, LiteLLM is provider-agnostic, so we might not need a hard exit here
# unless we are targeting a specific default model that requires a specific key.
# For now, we'll rely on LiteLLM's auto-detection.
# API_KEY = os.getenv("GOOGLE_API_KEY") # Or OPENAI_API_KEY, etc.
# if not API_KEY:
#     print("Error: Necessary API key (e.g., GOOGLE_API_KEY, OPENAI_API_KEY) not set or found.", file=sys.stderr)
#     print(f"Please ensure your API key is set in {dotenv_path}", file=sys.stderr)
#     sys.exit(1)

class LiteLLMMcpClient:
    """A client that connects to an MCP server and uses LiteLLM for interaction."""

    def __init__(self, system_instruction: Optional[str] = None):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Read model ID from .env, with a fallback default
        self.model_name = os.getenv("LITELLM_MODEL_ID", "gemini/gemini-2.5-pro-preview-05-06")
        
        self.system_instruction_content = system_instruction
        self.conversation_history: list[Dict[str, Any]] = []
        if self.system_instruction_content:
            self.conversation_history.append({"role": "system", "content": self.system_instruction_content})
        
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
                print(f"Warning: Could not create first run flag at {flag_path}: {e}", file=sys.stderr)
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
        # print("Database session state cleaned.")

    def _reset_feedback_cycle_state(self):
        """Resets state for a new natural language query and its feedback cycle."""
        # print("Resetting feedback cycle state for new SQL generation...") # Internal detail
        self.current_natural_language_question = None
        self.current_feedback_report_content = None

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
                print("Error: No tools discovered from the MCP server. Exiting.", file=sys.stderr)
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
                    print(f"Warning: Skipping MCP tool '{tool_name}' due to missing attributes for LiteLLM conversion.", file=sys.stderr) # Keep this warning
            
            if not self.litellm_tools and mcp_tools_list:
                print("Error: No MCP tools could be converted to LiteLLM format. Exiting.", file=sys.stderr)
                sys.exit(1)
            # print(f"LiteLLM client configured to use model '{self.model_name}' with {len(self.litellm_tools)} tools.") # Internal detail
        except Exception as e:
            print(f"Fatal error during MCP server connection or LiteLLM setup: {e}. Please ensure the MCP server script is correct and executable.", file=sys.stderr)
            await self.cleanup()
            sys.exit(1)

    async def _send_message_to_llm(self, messages: list, tools: Optional[list] = None, tool_choice: str = "auto") -> Any:
        """Sends messages to LiteLLM and handles response, including tool calls."""
        try:
            # print(f"Sending to LiteLLM ({self.model_name}). Messages: {json.dumps(messages, indent=2)[:200]}...") # Debug, too verbose
            # if tools: # Debug
            #     print(f"Tools provided: {[tool['function']['name'] for tool in tools]}")
            
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None # only if tools are present
            )
            # Add user message to history (assistant response added after processing)
            # The last message in `messages` is the current user prompt
            if messages[-1]["role"] == "user":
                 self.conversation_history.append(messages[-1])

            return response
        except Exception as e:
            print(f"Error calling LiteLLM: {e}", file=sys.stderr)
            # Add user message to history even if call fails, to keep track
            if messages[-1]["role"] == "user" and messages[-1] not in self.conversation_history:
                 self.conversation_history.append(messages[-1])
            # Add a placeholder error message to history for the assistant
            self.conversation_history.append({"role": "assistant", "content": f"Error communicating with LLM: {e}"})
            raise # Re-raise the exception to be handled by the caller

    async def _process_llm_response(self, llm_response: Any) -> Tuple[str, bool]:
        """Processes LiteLLM response, handles tool calls, and returns text response and if a tool was called."""
        assistant_response_content = ""
        tool_calls_made = False

        if not llm_response or not llm_response.choices or not llm_response.choices[0].message:
            assistant_response_content = "Error: Empty or invalid response from LLM."
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
            if self.current_natural_language_question != nl_question: 
                self._reset_feedback_cycle_state()
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

            return sql_gen_result_dict.get("message_to_user", "Error: No message from SQL generation.")

        # Command: /feedback: {user_feedback_text}
        feedback_match = re.match(r"/feedback:\s*(.+)", query, re.IGNORECASE)
        if feedback_match:
            if not self.current_feedback_report_content or not self.current_natural_language_question:
                return "Error: No SQL query generated yet for feedback. Use /generate_sql first."
            
            user_feedback_text = feedback_match.group(1).strip()
            # print(f"Received /feedback: {user_feedback_text}") # User sees the result
            print("Processing feedback, please wait...") # User-facing wait message

            current_report_json = self.current_feedback_report_content.model_dump_json(indent=2)
            # Get schema as dict, then dump to string with indent for the prompt
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
                    # Construct messages for LiteLLM
                    messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt}]
                    
                    # For this specific call, we expect a JSON response, not tool usage.
                    # So, we don't pass self.litellm_tools here.
                    # Some models might need specific instructions to output JSON (e.g. in the prompt or via response_format for OpenAI)
                    # LiteLLM's acompletion will pass response_format if the model supports it.
                    # We can try adding "response_format": {"type": "json_object"} if using a compatible OpenAI model.
                    # For Gemini, the prompt needs to be very clear about JSON output.
                    
                    llm_response_obj = await self._send_message_to_llm(messages_for_llm) # No tools for this specific JSON response
                    response_text, _ = await self._process_llm_response(llm_response_obj) # This will add to history

                    # response_text should be the JSON string
                    if response_text.startswith("```json"): response_text = response_text[7:]
                    if response_text.endswith("```"): response_text = response_text[:-3]
                    
                    updated_report_model = FeedbackReportContentModel.model_validate_json(response_text.strip())
                    
                    # Basic check for SELECT
                    if not updated_report_model.final_corrected_sql_query or \
                       not updated_report_model.final_corrected_sql_query.strip().upper().startswith("SELECT"):
                        raise ValueError("Corrected SQL in feedback report must start with SELECT.")

                    self.current_feedback_report_content = updated_report_model
                    
                    # Now, execute the new final_corrected_sql_query for verification
                    exec_result = None
                    exec_error = None
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
                    print(f"Error processing LLM feedback response (Attempt {attempt+1}): {e}")
                    if attempt == MAX_FEEDBACK_RETRIES:
                        # Use the raw response_text from the LLM if available and parsing failed
                        last_llm_resp_text_for_error = response_text if 'response_text' in locals() else "No response text captured."
                        return f"Error processing feedback after multiple attempts: {e}. Last LLM response: {last_llm_resp_text_for_error}"
                    feedback_prompt = f"Your previous response for updating the feedback report was invalid. Error: {e}. Please try again, ensuring the entire response is a single valid JSON object conforming to the FeedbackReportContentModel schema and SQL starts with SELECT."
                except Exception as e_gen: # Catch-all for other unexpected errors
                    print(f"Unexpected error during feedback processing (Attempt {attempt+1}): {e_gen}")
                    if attempt == MAX_FEEDBACK_RETRIES:
                        return f"Unexpected error processing feedback: {e_gen}."
                    # Generic retry for unexpected errors
                    feedback_prompt = "An unexpected error occurred. Please try to regenerate the updated feedback report JSON."


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
                    print("Warning: Failed to generate or update insights from this feedback.") # Keep this warning
                
                if self.current_natural_language_question and self.current_feedback_report_content.final_corrected_sql_query:
                    try:
                        memory_module.save_nl2sql_pair(
                            self.current_db_name_identifier,
                            self.current_natural_language_question,
                            self.current_feedback_report_content.final_corrected_sql_query
                        )
                        # print("NLQ-SQL pair saved.") # User sees final message
                    except Exception as e_nl2sql:
                        print(f"Warning: Failed to save NLQ-SQL pair: {e_nl2sql}") # Keep this warning
                else:
                    print("Warning: Could not save NLQ-SQL pair due to missing NLQ or final SQL query in the report.") # Keep this

                final_user_message = f"Approved. Feedback report, insights, and NLQ-SQL pair for '{self.current_db_name_identifier}' saved."
                if not insights_success:
                     final_user_message += " (Insights update may have failed, check logs)."
                
                self._reset_feedback_cycle_state()
                return f"{final_user_message}\nYou can start a new query with /generate_sql."

            except Exception as e:
                return f"Error during approval process: {e}"
        
        if query.strip().lower() == "/list_commands":
            return self._get_commands_help_text()

        if not query.startswith("/"):
            # print(f"Handling as navigation query: {query}") # User sees the result
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
            "      Example: /change_database: \"postgresql://postgres:pwd@localhost:5432/mydb\"\n"
            "  /generate_sql: {Your natural language question for SQL}\n"
            "    - Generates a SQL query based on your question.\n"
            "      Example: /generate_sql: Show me all active users from the 'customers' table\n"
            "  /feedback: {Your feedback on the last generated SQL}\n"
            "    - Provide feedback to refine the last SQL query.\n"
            "      Example: /feedback: The query is missing a WHERE clause for active status\n"
            "  /approved\n"
            "    - Approves the last SQL query. Saves the report, insights, and NLQ-SQL pair.\n"
            "  /list_commands\n"
            "    - Shows this list of available commands.\n"
            "  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights."
        )

    async def chat_loop(self):
        print("\nPostgreSQL Co-Pilot Started!")
        print(f"Using LiteLLM with model: {self.model_name}")
        
        is_first_run = self._check_and_set_first_run()
        if is_first_run:
            print("\nWelcome! It looks like this is your first time running the Co-Pilot.")
            print("Please ensure you have run 'python prerequisites.py' to set up necessary folders and get API key instructions.")
            print(self._get_commands_help_text()) # Show full help on first run
        else:
            # On subsequent runs, just remind them about /list_commands
            print("\nType '/list_commands' to see available commands.")
        
        print("\nType a PostgreSQL connection string (e.g., postgresql://user:pass@host:port/dbname) to begin,")
        print("or use '/change_database: \"your_connection_string\"'. Type 'quit' to exit.")

        while True:
            try:
                user_input_query = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input(f"\n[{self.current_db_name_identifier or 'No DB'}] Query: ").strip()
                )
                if user_input_query.lower() == 'quit':
                    await self._cleanup_database_session(full_cleanup=True)
                    break
                if not user_input_query: continue
                
                # Add user input to conversation history before dispatching
                # self.conversation_history.append({"role": "user", "content": user_input_query})
                # Dispatch command will handle adding to history via _send_message_to_llm

                response_text = await self.dispatch_command(user_input_query)
                print("\nAI: " + response_text)

            except EOFError: print("\nExiting chat loop."); break
            except Exception as e: print(f"\nError in chat loop: {str(e)}")

    async def cleanup(self):
        print("\nCleaning up client resources...")
        await self._cleanup_database_session(full_cleanup=True)
        await self.exit_stack.aclose()
        print("Client cleanup complete.")

async def main():
    # Set LiteLLM verbosity (optional)
    # litellm.set_verbose = True # or litellm.success_callback = [...] etc.
    # litellm.telemetry = False # Disable LiteLLM telemetry if desired

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')

    if len(sys.argv) < 2:
        print("Usage: python postgres_copilot_chat.py <path_to_mcp_server_script.py>", file=sys.stderr)
        print("Example: python postgres_copilot_chat.py ./postgresql_server.py", file=sys.stderr)
        sys.exit(1)
    
    mcp_server_script = sys.argv[1]
    if not os.path.exists(mcp_server_script):
        print(f"Error: MCP server script not found at '{mcp_server_script}'", file=sys.stderr)
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
    
    try:
        # memory_module.ensure_memory_directories() is called on its import.
        print("PostgreSQL Co-Pilot Client (LiteLLM) starting...")
        await client.connect_to_mcp_server(mcp_server_script) # Connect to the local postgresql_server.py
        
        # With LiteLLM, there's no explicit chat object to check like `client.chat`.
        # We assume connection is successful if connect_to_mcp_server doesn't exit.
        await client.chat_loop()

        # Save conversation history (now client.conversation_history)
        if client.conversation_history:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file_path = os.path.join(memory_module.CONVERSATION_HISTORY_DIR, f"conversation_litellm_{timestamp}.json")
            print(f"\nSaving conversation history to: {history_file_path}")
            try:
                with open(history_file_path, "w", encoding="utf-8") as f:
                    # Save as JSON for better structure, especially with tool calls
                    history_to_save = {
                        "db_name": client.current_db_name_identifier or "N/A",
                        "model_used": client.model_name,
                        "history": client.conversation_history
                    }
                    json.dump(history_to_save, f, indent=2)
                print("Conversation history saved.")
            except Exception as e_hist:
                print(f"Error saving conversation history: {e_hist}")
        else:
            print("No conversation history to save.")

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
