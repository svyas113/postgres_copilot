import asyncio
import os
import sys
import subprocess # Keep for StdioServerParameters if server is run as subprocess
from dotenv import load_dotenv
from typing import Optional, Any, Dict, Tuple
from contextlib import AsyncExitStack
import datetime
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GenAiTool
from pydantic import ValidationError # For catching Pydantic errors

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool 
import copy
import re
import json

# Import local modules
import memory_module # Handles all file/directory operations

# --- Directory and .env Setup ---
# Ensure the 'passwords' directory exists
PASSWORDS_DIR = os.path.join(os.path.dirname(__file__), 'passwords')
os.makedirs(PASSWORDS_DIR, exist_ok=True)
print(f"Ensured 'passwords' directory exists at: {PASSWORDS_DIR}")

# Construct the path to the .env file within the 'passwords' subdirectory
dotenv_path = os.path.join(PASSWORDS_DIR, '.env')

# Check if .env exists in 'passwords', if not, create a template
if not os.path.exists(dotenv_path):
    print(f".env file not found in {PASSWORDS_DIR}. Creating a template .env file...")
    try:
        with open(dotenv_path, 'w', encoding='utf-8') as f:
            f.write("# Please enter your Google API Key for Gemini\n")
            f.write("GOOGLE_API_KEY=\n")
        print(f"Template .env file created at: {dotenv_path}. Please fill in your GOOGLE_API_KEY.")
    except Exception as e:
        print(f"Error creating template .env file at {dotenv_path}: {e}", file=sys.stderr)
        # Continue, but API key loading will likely fail.
# --- End Directory and .env Setup ---

import initialization_module
import sql_generation_module
import insights_module
import database_navigation_module
from pydantic_models import SQLGenerationResponse, FeedbackReportContentModel, FeedbackIteration #, InsightsExtractionModel

# Load environment variables from .env file
# dotenv_path is already defined above
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load .env from: {dotenv_path}")
    # Check if GOOGLE_API_KEY was actually loaded
    API_KEY_CHECK = os.getenv("GOOGLE_API_KEY")
    if API_KEY_CHECK:
        print(f"Successfully loaded GOOGLE_API_KEY from {dotenv_path}.")
    else:
        print(f"Loaded {dotenv_path}, but GOOGLE_API_KEY was not found within it or is empty.")
else:
    # This case should ideally not be hit if the template creation worked,
    # but kept as a fallback message.
    print(f".env file not found at {dotenv_path}. GOOGLE_API_KEY will likely be missing.")
    # We don't call load_dotenv() without a path here, as we specifically want it from PASSWORDS_DIR

# Configure the Google Generative AI library
API_KEY = os.getenv("GOOGLE_API_KEY") # This will be None if not set or file not found/empty
if not API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set or found in the .env file.", file=sys.stderr)
    print(f"Please ensure your GOOGLE_API_KEY is set in {dotenv_path}", file=sys.stderr)
    sys.exit(1)
genai.configure(api_key=API_KEY)

class GeminiMcpClient:
    """A client that connects to an MCP server and uses the Gemini model for interaction."""

    def __init__(self, system_instruction: Optional[str] = None):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.chat = None
        self.system_information = system_instruction
        
        # Ensure memory directories are set up by memory_module on import
        # self.memory_module will be the imported module itself.
        print(f"Base memory directory: {memory_module.BASE_MEMORY_DIR}")
        print(f"Feedback directory: {memory_module.FEEDBACK_DIR}")
        print(f"Insights directory: {memory_module.INSIGHTS_DIR}")
        print(f"Schema directory: {memory_module.SCHEMA_DIR}")
        print(f"Conversation history directory: {memory_module.CONVERSATION_HISTORY_DIR}")

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

    async def _cleanup_database_session(self, full_cleanup: bool = True):
        print("Cleaning up database session state...")
        if full_cleanup:
            self.is_initialized = False
            self.current_db_connection_string = None
            self.current_db_name_identifier = None
            self.db_schema_and_sample_data = None
            self.cumulative_insights_content = None # Cleared when DB changes
        
        # Always reset feedback-cycle specific state
        self._reset_feedback_cycle_state()
        print("Database session state cleaned.")

    def _reset_feedback_cycle_state(self):
        """Resets state for a new natural language query and its feedback cycle."""
        print("Resetting feedback cycle state for new SQL generation...")
        self.current_natural_language_question = None
        self.current_feedback_report_content = None
        # Any other temporary states related to a single query cycle would be reset here.

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
        print(f"Connecting to MCP server script: {server_script_path}")
        # ... (MCP server connection logic - largely unchanged from previous versions) ...
        # This part sets up self.session and self.chat with tools from the MCP server
        is_python = server_script_path.endswith('.py')
        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.session.initialize()
            response = await self.session.list_tools()
            mcp_tools: list[McpTool] = response.tools
            print(f"Connected to MCP server with {len(mcp_tools)} tools: {[tool.name for tool in mcp_tools]}")
            if not mcp_tools: sys.exit("No tools discovered from the MCP server. Exiting.")

            gemini_tools = []
            for mcp_tool_obj in mcp_tools:
                # Manual conversion to Gemini FunctionDeclaration
                # (Ensure attributes like name, description, inputSchema are correct)
                tool_name = getattr(mcp_tool_obj, 'name', None)
                tool_desc = getattr(mcp_tool_obj, 'description', None)
                tool_schema = copy.deepcopy(getattr(mcp_tool_obj, 'inputSchema', {})) # Deep copy
                if tool_schema: # Sanitize schema
                    tool_schema.pop('title', None)
                    if 'properties' in tool_schema and isinstance(tool_schema['properties'], dict):
                        for prop_details in tool_schema['properties'].values():
                            if isinstance(prop_details, dict): prop_details.pop('title', None)
                    if 'properties' in tool_schema and 'type' not in tool_schema: tool_schema['type'] = 'object'

                if tool_name and tool_desc:
                    func_decl = FunctionDeclaration(name=tool_name, description=tool_desc, parameters=tool_schema or None)
                    gemini_tools.append(GenAiTool(function_declarations=[func_decl]))
                else: print(f"Warning: Skipping MCP tool '{tool_name}' due to missing attributes for Gemini conversion.", file=sys.stderr)
            
            if not gemini_tools and mcp_tools: sys.exit("Error: No MCP tools could be converted to Gemini format. Exiting.")

            try:
                # Try with gemini-1.0-pro first (newer version)
                print("Attempting to initialize Gemini model with 'gemini-2.5-pro-preview-05-06'...")
                model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06', tools=gemini_tools, system_instruction=self.system_information)
                self.chat = model.start_chat(enable_automatic_function_calling=False) # Manual tool calling
                print("Gemini chat session started with MCP tools integrated using gemini-2.5-pro-preview-05-06.")
            except Exception as model_error:
                print(f"Error initializing gemini-2.5-pro: {model_error}. Trying gemini-pro...")
                try:
                    # Fall back to gemini-pro
                    model = genai.GenerativeModel('gemini-pro', tools=gemini_tools, system_instruction=self.system_information)
                    self.chat = model.start_chat(enable_automatic_function_calling=False) # Manual tool calling
                    print("Gemini chat session started with MCP tools integrated using gemini-pro.")
                except Exception as fallback_error:
                    raise Exception(f"Failed to initialize Gemini models. First error: {model_error}, Fallback error: {fallback_error}")
        except Exception as e:
            print(f"Fatal error during MCP server connection or Gemini setup: {e}", file=sys.stderr)
            await self.cleanup()
            sys.exit(1)


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
            # Load cumulative insights for this DB if they exist
            self.cumulative_insights_content = memory_module.read_insights_file() # This reads the single insights file
            print(f"Initialization successful for {db_name_id}. Insights loaded (if any).")
            return message
        else:
            await self._cleanup_database_session(full_cleanup=True) # Ensure clean state on failure
            return message

    async def dispatch_command(self, query: str) -> str:
        if not self.chat or not self.session: # Should be caught by connect_to_mcp_server if it fails
            return "Error: Client not fully initialized (chat or session missing)."

        # Command: /change_database: "connection_string" or just "connection_string" for implicit init
        conn_str_explicit_match = re.match(r"/change_database:\s*\"(.+)\"", query, re.IGNORECASE)
        parsed_conn_str, parsed_db_name_id = None, None

        if conn_str_explicit_match:
            raw_conn_str = conn_str_explicit_match.group(1)
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(raw_conn_str)
            if not parsed_conn_str: return "Error: Invalid connection string format provided with /change_database."
            print(f"Received /change_database for {parsed_db_name_id}...")
            return await self._handle_initialization(parsed_conn_str, parsed_db_name_id)
        
        # Implicit initialization if not initialized and query looks like a connection string
        if not self.is_initialized:
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(query)
            if parsed_conn_str:
                print(f"Attempting implicit initialization for {parsed_db_name_id}...")
                return await self._handle_initialization(parsed_conn_str, parsed_db_name_id)
            else:
                return "Error: Database not initialized. Please provide a connection string or use /change_database."

        # --- At this point, database should be initialized ---
        if not self.is_initialized or not self.current_db_name_identifier or self.db_schema_and_sample_data is None:
             # This state should ideally not be reached if logic above is correct
            return "Critical Error: Database initialization state is inconsistent. Please try /change_database again."

        # Command: /generate_sql: {Natural language question}
        generate_sql_match = re.match(r"/generate_sql:\s*(.+)", query, re.IGNORECASE)
        if generate_sql_match:
            nl_question = generate_sql_match.group(1).strip()
            if self.current_natural_language_question != nl_question: # New question
                self._reset_feedback_cycle_state()
            self.current_natural_language_question = nl_question
            print(f"Received /generate_sql for: {nl_question}")

            # Call sql_generation_module
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
            print(f"Received /feedback: {user_feedback_text}")

            # Prepare prompt for LLM to generate corrected SQL and update the FeedbackReportContentModel
            # The LLM needs the current state of self.current_feedback_report_content
            current_report_json = self.current_feedback_report_content.model_dump_json(indent=2)
            feedback_model_schema = FeedbackReportContentModel.model_json_schema(indent=2)

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
                f"```json\n{feedback_model_schema}\n```\n"
                f"Ensure the `corrected_sql_attempt` and `final_corrected_sql_query` start with SELECT."
            )
            
            MAX_FEEDBACK_RETRIES = 1
            for attempt in range(MAX_FEEDBACK_RETRIES + 1):
                try:
                    llm_response = await self.chat.send_message_async(feedback_prompt)
                    response_text = "".join(part.text for part in llm_response.parts if hasattr(part, 'text') and part.text).strip()
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
                        return f"Error processing feedback after multiple attempts: {e}. Last LLM response: {response_text}"
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

            print("Updating memory, please wait a few seconds...")
            try:
                # 1. Save Feedback Markdown
                feedback_filepath = memory_module.save_feedback_markdown(
                    self.current_feedback_report_content, 
                    self.current_db_name_identifier
                )
                print(f"Feedback report saved to: {feedback_filepath}")

                # 2. Read the saved feedback markdown for insights generation
                saved_feedback_md_content = memory_module.read_feedback_file(feedback_filepath)
                if not saved_feedback_md_content: # Should not happen if save was successful
                    return "Error: Could not read back saved feedback file for insights processing."

                # 3. Generate and Update Insights
                insights_success = await insights_module.generate_and_update_insights(
                    self, # Pass the client instance for LLM calls
                    saved_feedback_md_content,
                    self.current_db_name_identifier
                )
                if insights_success:
                    print("Insights successfully generated/updated.")
                    # Update in-memory insights
                    self.cumulative_insights_content = memory_module.read_insights_file()
                else:
                    print("Warning: Failed to generate or update insights from this feedback.")
                    # Continue, as feedback is saved, but inform user.

                final_user_message = f"Feedback report saved to {feedback_filepath}."
                if insights_success: final_user_message += " Cumulative insights updated."
                else: final_user_message += " Insights update failed (see logs)."
                
                self._reset_feedback_cycle_state()
                return f"{final_user_message}\nYou can start a new query with /generate_sql."

            except Exception as e:
                return f"Error during approval process: {e}"
        
        # Command: /list_commands
        if query.strip().lower() == "/list_commands":
            commands_help_text = (
                "Available commands:\n"
                "  /change_database: \"postgresql://user:pass@host:port/dbname\"\n"
                "    - Connects to a new PostgreSQL database or changes the active one.\n"
                "      The part after the last '/' in the connection string will be used as the database identifier.\n"
                "  /generate_sql: {Natural language question to generate SQL}\n"
                "    - Generates a SQL query based on your natural language question.\n"
                "      Example: /generate_sql: Show me all active users\n"
                "  /feedback: {Your feedback on the generated SQL}\n"
                "    - Provide feedback on the last generated SQL query to refine it.\n"
                "      Example: /feedback: The query is missing a WHERE clause for active status\n"
                "  /approved\n"
                "    - Approves the last generated/corrected SQL query. This saves the feedback report and updates insights.\n"
                "  /list_commands\n"
                "    - Shows this list of available commands.\n"
                "  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights."
            )
            return commands_help_text

        # Fallback: Database Navigation (no '/' prefix)
        if not query.startswith("/"):
            print(f"Handling as navigation query: {query}")
            return await database_navigation_module.handle_navigation_query(
                self, query, self.current_db_name_identifier, 
                self.cumulative_insights_content, self.db_schema_and_sample_data
            )

        return (f"Unknown command or query format. Current database: {self.current_db_name_identifier}.\n"
                f"Available commands: /change_database, /generate_sql, /feedback, /approved, /list_commands, or type a question to navigate.")

    async def chat_loop(self):
        print("\nPostgreSQL Co-Pilot Started!")
        print("Type a PostgreSQL connection string (e.g., postgresql://user:pass@host:port/dbname) to begin,")
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
    )
    client = GeminiMcpClient(system_instruction=custom_system_prompt)
    
    try:
        # memory_module.ensure_memory_directories() is called on its import.
        print("PostgreSQL Co-Pilot Client starting...")
        await client.connect_to_mcp_server(mcp_server_script) # Connect to the local postgresql_server.py
        
        if client.chat: # Check if chat was successfully initialized
            await client.chat_loop()

            # Save conversation history
            if client.chat and client.chat.history: # Check history again, just in case
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                history_file_path = os.path.join(memory_module.CONVERSATION_HISTORY_DIR, f"conversation_{timestamp}.txt")
                print(f"\nSaving conversation history to: {history_file_path}")
                try:
                    with open(history_file_path, "w", encoding="utf-8") as f:
                        f.write(f"--- Conversation History for DB: {client.current_db_name_identifier or 'N/A'} ---\n")
                        if client.system_information: f.write(f"System: {client.system_information}\n")
                        for entry in client.chat.history:
                            role = entry.role
                            text_parts = [part.text for part in entry.parts if hasattr(part, 'text') and part.text]
                            # TODO: Add more sophisticated history logging for tool calls/responses if needed
                            if text_parts: f.write(f"{role.capitalize()}: {' '.join(text_parts)}\n")
                    print("Conversation history saved.")
                except Exception as e_hist: print(f"Error saving conversation history: {e_hist}")
            else: print("No conversation history to save.")
        else:
            print("Could not start chat loop as Gemini chat session was not initialized.", file=sys.stderr)

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
