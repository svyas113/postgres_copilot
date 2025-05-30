import sys
import logging
import os
from datetime import datetime

# Import necessary components from the mcp SDK
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filename='file_management_server.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
mcp_logger = logging.getLogger('mcp')
mcp_logger.setLevel(logging.DEBUG)
if not mcp_logger.handlers:
    file_handler = logging.FileHandler('file_management_server.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    mcp_logger.addHandler(file_handler)

logger.info("File Management MCP Server script started.")

# Define base paths for folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
INSIGHTS_DIR = os.path.join(BASE_DIR, "insights")
CONVERSATION_HISTORY_DIR = os.path.join(BASE_DIR, "conversation_history")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
INSIGHTS_FILE_PATH = os.path.join(INSIGHTS_DIR, "insights.txt")

# Define the main sections for insights.txt
INSIGHTS_FILE_TOP_HEADER = "# LLM Generated Insights"
INSIGHTS_SECTIONS = {
    "Schema Understanding": "## Schema Understanding",
    "Query Construction Best Practices": "## Query Construction Best Practices",
    "Common Errors and Corrections": "## Common Errors and Corrections"
}
# Ordered list of section headers for initialization and finding next section
ORDERED_INSIGHTS_HEADERS = list(INSIGHTS_SECTIONS.values())


# Create directories if they don't exist
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(INSIGHTS_DIR, exist_ok=True)
os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

logger.info(f"Feedback directory: {FEEDBACK_DIR}")
logger.info(f"Insights directory: {INSIGHTS_DIR}")
logger.info(f"Conversation history directory: {CONVERSATION_HISTORY_DIR}")
logger.info(f"Memory directory: {MEMORY_DIR}")

# Create a FastMCP server instance
logger.info("Creating FastMCP instance for File Management server...")
mcp = FastMCP("FileManagementServer")
logger.info("FastMCP instance created.")

def _initialize_insights_file():
    """
    Ensures the insights file exists and contains all predefined section headers.
    If the file doesn't exist or is empty, it's created with the full structure.
    If it exists but is missing sections, missing sections are appended.
    """
    if not os.path.exists(INSIGHTS_FILE_PATH) or os.path.getsize(INSIGHTS_FILE_PATH) == 0:
        logger.info(f"Initializing insights file with full structure: {INSIGHTS_FILE_PATH}")
        with open(INSIGHTS_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(f"{INSIGHTS_FILE_TOP_HEADER}\n\n")
            for header_text in ORDERED_INSIGHTS_HEADERS:
                f.write(f"{header_text}\n\n")
        return

    # If file exists, check and append missing sections
    try:
        with open(INSIGHTS_FILE_PATH, "r+", encoding="utf-8") as f:
            content = f.read()
            # Check if all headers are present, if not, prepare to append them
            text_to_append_for_missing_sections = ""
            current_file_content_for_check = content # Use this to check if a header is already in file or will be appended

            for header_key in INSIGHTS_SECTIONS: # Iterate in defined order
                header_text = INSIGHTS_SECTIONS[header_key]
                if header_text not in current_file_content_for_check:
                    logger.info(f"'{header_text}' not found. Will append to {INSIGHTS_FILE_PATH}")
                    # Add spacing before appending new section if needed
                    if current_file_content_for_check.strip() and not current_file_content_for_check.endswith("\n\n"):
                        if not current_file_content_for_check.endswith("\n"):
                            text_to_append_for_missing_sections += "\n"
                        text_to_append_for_missing_sections += "\n"
                    
                    text_to_append_for_missing_sections += f"{header_text}\n\n"
                    current_file_content_for_check += text_to_append_for_missing_sections # So next check sees it

            if text_to_append_for_missing_sections:
                f.seek(0, os.SEEK_END) # Go to end of file
                f.write(text_to_append_for_missing_sections)
                logger.info(f"Updated insights file by appending missing sections: {INSIGHTS_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error during insights file initialization/check: {e}", exc_info=True)


@mcp.tool()
def write_feedback_file(content: str) -> str:
    """
    Writes the given content to a new time-stamped .txt file in the 'feedback' folder.
    """
    logger.debug("Tool 'write_feedback_file' called.")
    _initialize_insights_file() # Ensure insights structure is fine, though not directly used here
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"feedback_{timestamp}.txt"
        filepath = os.path.join(FEEDBACK_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote feedback to {filepath}")
        return f"Successfully wrote feedback to {filepath}"
    except Exception as e:
        logger.error(f"Failed to write feedback file: {e}", exc_info=True)
        return f"Error: Failed to write feedback file. {e}"

@mcp.tool()
def write_insights_file(content: str) -> str:
    """
    DEPRECATED. Use 'edit_insights_file' to add content to specific sections.
    This tool overwrites 'insights.txt' with the given content.
    It's generally recommended to use 'edit_insights_file' instead.
    """
    logger.warning("Tool 'write_insights_file' called. This tool overwrites the entire insights file.")
    _initialize_insights_file() # Ensure insights structure is fine before overwriting (or re-init after)
    try:
        with open(INSIGHTS_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        # After a full write, it's good to ensure our structure is still somewhat present or re-initialize
        _initialize_insights_file() # This will add missing sections if the written content didn't include them
        logger.info(f"Successfully overwrote insights file: {INSIGHTS_FILE_PATH}")
        return f"Successfully overwrote insights file: {INSIGHTS_FILE_PATH}. Consider using 'edit_insights_file'."
    except Exception as e:
        logger.error(f"Failed to write insights file: {e}", exc_info=True)
        return f"Error: Failed to write insights file. {e}"

@mcp.tool()
def edit_insights_file(section_name: str, content_to_add: str) -> str:
    """
    Appends content under a specific section in 'insights.txt'.
    Valid section_names are: "Schema Understanding", "Query Construction Best Practices", "Common Errors and Corrections".
    The content_to_add should be formatted as desired (e.g., markdown bullet points).
    """
    logger.debug(f"Tool 'edit_insights_file' called for section: '{section_name}'")

    if section_name not in INSIGHTS_SECTIONS:
        logger.warning(f"Invalid section_name '{section_name}' for edit_insights_file.")
        valid_sections = ", ".join(INSIGHTS_SECTIONS.keys())
        return f"Error: Invalid section_name. Must be one of: {valid_sections}."

    _initialize_insights_file() # Ensure file and basic headers exist

    target_header_text = INSIGHTS_SECTIONS[section_name]
    processed_content_to_add = content_to_add.strip()

    if not processed_content_to_add:
        logger.info("Content to add is empty. No changes made to insights file.")
        return "Info: Content to add was empty. No changes made."

    try:
        with open(INSIGHTS_FILE_PATH, "r+", encoding="utf-8") as f:
            lines = f.readlines() # Read all lines, they include '\n'

            target_header_line_idx = -1
            for i, line_content in enumerate(lines):
                if line_content.strip() == target_header_text:
                    target_header_line_idx = i
                    break
            
            if target_header_line_idx == -1:
                # This should ideally be handled by _initialize_insights_file appending missing sections.
                # If it still occurs, it's an issue with initialization or file state.
                logger.error(f"Critical: Section header '{target_header_text}' not found in {INSIGHTS_FILE_PATH} even after initialization attempt.")
                return f"Error: Could not find or ensure section header '{target_header_text}' in the insights file. Please check file integrity."

            # Determine where the target section ends. This is the start of the *next* known header,
            # or any "## " style header, or EOF.
            insertion_point_idx = len(lines) # Default to end of file

            # Find the order of the current target header
            current_target_order_idx = -1
            try:
                current_target_order_idx = ORDERED_INSIGHTS_HEADERS.index(target_header_text)
            except ValueError:
                pass # Should not happen if target_header_text is from INSIGHTS_SECTIONS

            # Look for the next *known* header first
            if current_target_order_idx != -1:
                for i in range(current_target_order_idx + 1, len(ORDERED_INSIGHTS_HEADERS)):
                    next_known_header = ORDERED_INSIGHTS_HEADERS[i]
                    for line_num in range(target_header_line_idx + 1, len(lines)):
                        if lines[line_num].strip() == next_known_header:
                            insertion_point_idx = line_num
                            break
                    if insertion_point_idx != len(lines): # Found it
                        break
            
            # If no subsequent *known* header was found, look for any generic "## " header
            if insertion_point_idx == len(lines):
                for line_num in range(target_header_line_idx + 1, len(lines)):
                    if lines[line_num].strip().startswith("## ") and lines[line_num].strip() != target_header_text:
                        insertion_point_idx = line_num
                        break
            
            # Now, `insertion_point_idx` is the line number *before which* we want to insert.
            # This is either the start of the next section or the end of the file.

            # Prepare the actual string to insert
            string_block_to_insert = ""
            # Add a leading newline if the line just before the insertion point
            # (which is the last line of the current section's content, or the header itself)
            # is not empty and is not the header itself.
            if insertion_point_idx > target_header_line_idx + 1 and \
               lines[insertion_point_idx - 1].strip() != "":
                string_block_to_insert += "\n" # Add a blank line for separation

            string_block_to_insert += processed_content_to_add + "\n" # Add the content, ensure it ends with a newline

            # Add a trailing blank line if there's a next section header right after our insertion.
            # This ensures: Content\n\n## Next Header
            if insertion_point_idx < len(lines) and \
               lines[insertion_point_idx].strip().startswith("## ") and \
               not string_block_to_insert.endswith("\n\n"):
                string_block_to_insert += "\n"
            # If inserting at EOF, also ensure a final blank line for consistency if not already there
            elif insertion_point_idx == len(lines) and not string_block_to_insert.endswith("\n\n"):
                 string_block_to_insert += "\n"


            # Reconstruct the file content
            final_lines = lines[:insertion_point_idx]
            # Split the block by lines to insert them properly
            for content_line in string_block_to_insert.splitlines(keepends=True):
                final_lines.append(content_line)
            final_lines.extend(lines[insertion_point_idx:])
            
            f.seek(0)
            f.writelines(final_lines)
            f.truncate()
        
        logger.info(f"Successfully edited insights file: {INSIGHTS_FILE_PATH}, added to section: '{section_name}'")
        return f"Successfully added content to section '{section_name}' in {INSIGHTS_FILE_PATH}."

    except Exception as e:
        logger.error(f"Failed to edit insights file under section '{section_name}': {e}", exc_info=True)
        return f"Error: Failed to edit insights file. {e}"


@mcp.tool()
def read_feedback_file(filename: str) -> str:
    """
    Reads the content of a specified .txt file from the 'feedback' folder.
    Provide only the filename (e.g., 'feedback_20250518_103000_123456.txt').
    """
    logger.debug(f"Tool 'read_feedback_file' called for filename: {filename}")
    try:
        # Sanitize filename to prevent directory traversal
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            logger.error(f"Invalid characters in filename: {filename}")
            return "Error: Invalid filename."
        
        filepath = os.path.join(FEEDBACK_DIR, os.path.basename(filename)) # Use basename for added safety
        
        if not os.path.exists(filepath):
            logger.warning(f"Feedback file not found: {filepath}")
            return f"Error: Feedback file '{filename}' not found in {FEEDBACK_DIR}."
        # Double check it's still within the intended directory (paranoid check)
        if not os.path.abspath(filepath).startswith(os.path.abspath(FEEDBACK_DIR)):
            logger.error(f"Attempt to access file outside feedback directory: {filepath}")
            return "Error: Invalid file path."

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully read feedback file: {filepath}")
        return content
    except Exception as e:
        logger.error(f"Failed to read feedback file {filename}: {e}", exc_info=True)
        return f"Error: Failed to read feedback file {filename}. {e}"

@mcp.tool()
def read_insights_file() -> str:
    """
    Reads the content of the 'insights.txt' file from the 'insights' folder.
    """
    logger.debug("Tool 'read_insights_file' called.")
    _initialize_insights_file() # Ensure file exists before trying to read
    try:
        if not os.path.exists(INSIGHTS_FILE_PATH):
            # _initialize_insights_file should create it, but as a fallback:
            logger.warning(f"Insights file not found even after init: {INSIGHTS_FILE_PATH}")
            return f"Info: Insights file is empty or not yet created. ({INSIGHTS_FILE_PATH})"
        with open(INSIGHTS_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully read insights file: {INSIGHTS_FILE_PATH}")
        return content
    except Exception as e:
        logger.error(f"Failed to read insights file: {e}", exc_info=True)
        return f"Error: Failed to read insights file. {e}"

@mcp.tool()
def read_conversation_history_file(filename: str) -> str:
    """
    Reads the content of a specified file from the 'conversation_history' folder.
    Provide only the filename (e.g., 'chat_log_20250518.txt').
    """
    logger.debug(f"Tool 'read_conversation_history_file' called for filename: {filename}")
    try:
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            logger.error(f"Invalid characters in filename: {filename}")
            return "Error: Invalid filename."

        filepath = os.path.join(CONVERSATION_HISTORY_DIR, os.path.basename(filename))
        
        if not os.path.exists(filepath):
            logger.warning(f"Conversation history file not found: {filepath}")
            return f"Error: Conversation history file '{filename}' not found in {CONVERSATION_HISTORY_DIR}."
        if not os.path.abspath(filepath).startswith(os.path.abspath(CONVERSATION_HISTORY_DIR)):
            logger.error(f"Attempt to access file outside conversation_history directory: {filepath}")
            return "Error: Invalid file path."

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Successfully read conversation history file: {filepath}")
        return content
    except Exception as e:
        logger.error(f"Failed to read conversation history file {filename}: {e}", exc_info=True)
        return f"Error: Failed to read conversation history file {filename}. {e}"

@mcp.tool()
def get_memory_folder_path() -> str:
    """
    Returns the full absolute path to the 'memory' folder.
    """
    logger.debug("Tool 'get_memory_folder_path' called.")
    try:
        path = os.path.abspath(MEMORY_DIR)
        logger.info(f"Returning memory folder path: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to get memory folder path: {e}", exc_info=True)
        return f"Error: Failed to get memory folder path. {e}"

logger.info("File Management MCP tools defined.")

if __name__ == "__main__":
    logger.info("__main__ block started for File Management server.")
    # Initial check/creation of insights file structure on startup
    _initialize_insights_file()

    if sys.platform == "win32":
        logger.info("Configuring stdout/stdin for win32.")
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stdin.reconfigure(encoding='utf-8')
            logger.info("stdout/stdin configured for win32.")
        except Exception as e:
            logger.error(f"Error reconfiguring stdio for win32: {e}")


    logger.info("Starting MCP File Management Server (stdio)...")
    print("Starting MCP File Management Server (stdio)...", flush=True)

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.critical(f"Critical error running MCP File Management server: {e}", exc_info=True)
    finally:
        logger.info("MCP File Management server finished or exited.")
        logging.shutdown()