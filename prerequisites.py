import os
import sys
import subprocess

# --- Directory and File Creation ---
PASSWORDS_DIR_NAME = "passwords"
ENV_FILE_NAME = ".env"
MEMORY_DIR_NAME = "memory"
SUB_MEMORY_DIRS = ["feedback", "insights", "schema", "conversation_history", "NL2SQL"]

BASE_APP_DIR = os.path.dirname(os.path.abspath(__file__))

def create_passwords_dir_and_env_file():
    """Creates the 'passwords' directory and a template .env file within it."""
    passwords_path = os.path.join(BASE_APP_DIR, PASSWORDS_DIR_NAME)
    env_file_path = os.path.join(passwords_path, ENV_FILE_NAME)

    try:
        os.makedirs(passwords_path, exist_ok=True)
        print(f"Directory '{passwords_path}' ensured.")

        if not os.path.exists(env_file_path):
            with open(env_file_path, "w", encoding="utf-8") as f:
                f.write("# --- LiteLLM Model Configuration ---\n")
                f.write("# Specify the model ID you want LiteLLM to use.\n")
                f.write("# Default is set to a Gemini model, but you can change this to any LiteLLM supported model.\n")
                f.write("# Examples: 'openai/gpt-4', 'anthropic/claude-2', 'ollama/mistral', 'bedrock/anthropic.claude-v2'\n")
                f.write("LITELLM_MODEL_ID=gemini/gemini-2.5-pro-preview-05-06\n\n")

                f.write("# --- API Key Configuration ---\n")
                f.write("# Enter the API key for your chosen model provider.\n")
                f.write("# LiteLLM will automatically use the relevant key based on the LITELLM_MODEL_ID.\n\n")
                
                f.write("# For Gemini models (used by default LITELLM_MODEL_ID):\n")
                f.write("GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE\n\n")
                
                f.write("# For OpenAI models (e.g., if LITELLM_MODEL_ID is 'openai/gpt-4'):\n")
                f.write("# OPENAI_API_KEY=\n\n")
                
                f.write("# For Anthropic models (e.g., if LITELLM_MODEL_ID is 'anthropic/claude-2'):\n")
                f.write("# ANTHROPIC_API_KEY=\n\n")

                f.write("# --- Other Provider Authentication (e.g., AWS Bedrock, Azure OpenAI) ---\n")
                f.write("# For providers like AWS Bedrock or Azure OpenAI, LiteLLM uses standard SDK environment variables.\n")
                f.write("# Example for AWS Bedrock (ensure AWS CLI is configured or these are set):\n")
                f.write("# AWS_ACCESS_KEY_ID=\n")
                f.write("# AWS_SECRET_ACCESS_KEY=\n")
                f.write("# AWS_REGION_NAME=\n\n")
                
                f.write("# Example for Azure OpenAI (if LITELLM_MODEL_ID is 'azure/your-deployment-name'):\n")
                f.write("# AZURE_API_KEY=\n")
                f.write("# AZURE_API_BASE=\n")
                f.write("# AZURE_API_VERSION=\n\n")
                
                f.write("# For other providers, please refer to LiteLLM documentation for required environment variables.\n")

            print(f"Template '{ENV_FILE_NAME}' created in '{passwords_path}'.")
            print(f"IMPORTANT: Please open '{env_file_path}' and configure LITELLM_MODEL_ID and the corresponding API key(s) or authentication variables.")
        else:
            print(f"'{ENV_FILE_NAME}' already exists in '{passwords_path}'. Please ensure it's configured correctly for LiteLLM.")
    except Exception as e:
        print(f"Error creating passwords directory or .env file: {e}", file=sys.stderr)
        return False
    return True

def create_memory_directories():
    """Creates the 'memory' directory and its subdirectories."""
    memory_base_path = os.path.join(BASE_APP_DIR, MEMORY_DIR_NAME)
    try:
        os.makedirs(memory_base_path, exist_ok=True)
        print(f"Base memory directory '{memory_base_path}' ensured.")
        for sub_dir in SUB_MEMORY_DIRS:
            os.makedirs(os.path.join(memory_base_path, sub_dir), exist_ok=True)
        print(f"All subdirectories within '{memory_base_path}' ensured: {', '.join(SUB_MEMORY_DIRS)}")
    except Exception as e:
        print(f"Error creating memory directories: {e}", file=sys.stderr)
        return False
    return True

PYTHON_ALIAS = "python" if os.name != 'nt' else "python"

def check_pip_and_venv():
    """Checks if pip and venv are accessible."""
    try:
        subprocess.check_call([PYTHON_ALIAS, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pip is available.")
    except subprocess.CalledProcessError:
        print("Error: pip is not available or not found in PATH. Please ensure Python and pip are correctly installed.", file=sys.stderr)
        return False
    try:
        subprocess.check_call([PYTHON_ALIAS, "-m", "venv", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("venv module is available.")
    except subprocess.CalledProcessError:
        print("Error: Python 'venv' module is not available. Please install it (e.g., 'sudo apt-get install python3-venv' on Debian/Ubuntu or ensure it's part of your Python installation).", file=sys.stderr)
        return False
    return True

def display_setup_instructions():
    """Displays setup instructions for the user."""
    print("\n--- PostgreSQL Co-Pilot Setup Instructions ---")
    print("\n1. Python Virtual Environment (Recommended):")
    print("   It's highly recommended to use a Python virtual environment to manage dependencies.")
    print(f"   a. Create a virtual environment (e.g., in the '{os.path.basename(BASE_APP_DIR)}' directory):")
    print(f"      cd {BASE_APP_DIR}")
    print(f"      {PYTHON_ALIAS} -m venv .venv")
    print(f"   b. Activate the virtual environment:")
    if os.name == 'nt':
        print(f"      .\\.venv\\Scripts\\activate")
    else:
        print(f"      source .venv/bin/activate")
    print("      (You should see '(.venv)' at the beginning of your command prompt.)")

    print("\n2. Install Dependencies:")
    print(f"   Once your virtual environment is activated, install the required packages.")
    print(f"   The primary dependencies are listed in 'pyproject.toml'. You can install them using pip:")
    print(f"      pip install .  # This installs the project and its dependencies from pyproject.toml")
    print(f"   Alternatively, if a 'requirements.txt' is provided or generated, use:")
    print(f"      pip install -r requirements.txt")
    print(f"   Key dependencies include: litellm, python-dotenv, psycopg2-binary (or asyncpg), mcp.py (if local/pip).")

    print("\n3. Configure LLM Model and API Key:")
    env_file_display_path = os.path.join(PASSWORDS_DIR_NAME, ENV_FILE_NAME) # Relative path for display
    print(f"   Open the file '{env_file_display_path}' (located in '{os.path.join(BASE_APP_DIR, PASSWORDS_DIR_NAME)}')")
    print(f"   a. Set `LITELLM_MODEL_ID` to your desired model (e.g., 'openai/gpt-4', 'gemini/gemini-pro', 'ollama/mistral').")
    print(f"      The default is 'gemini/gemini-2.5-pro-preview-05-06'.")
    print(f"   b. Provide the corresponding API key (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`).")
    print(f"   c. For other providers like AWS Bedrock or Azure, set their specific environment variables as commented in the .env file.")
    print(f"      LiteLLM will use these environment variables for authentication.")

    print("\n4. Running the Application:")
    print(f"   Ensure your PostgreSQL server is running and accessible.")
    print(f"   Navigate to the application directory ('{BASE_APP_DIR}') in your terminal (if not already there).")
    print(f"   Make sure your virtual environment is activated.")
    print(f"   Run the main chat application using the command:")
    print(f"     {PYTHON_ALIAS} postgres_copilot_chat.py postgresql_server.py")
    print("   The `postgresql_server.py` is the MCP server script that interacts with your database.")

    print("\n--- Initial Commands ---")
    print("Once the application starts, you'll see '[No DB] Query:'.")
    print("To begin, you need to connect to your PostgreSQL database.")
    print("Provide your connection string directly, for example:")
    print("  postgresql://your_user:your_password@your_host:your_port/your_database_name")
    print("Or use the command:")
    print("  /change_database: \"postgresql://your_user:your_password@your_host:your_port/your_database_name\"")
    
    print("\nAvailable commands once connected (type /list_commands in the app for the full list):")
    print("  /generate_sql: {Your natural language question for SQL}")
    print("  /feedback: {Your feedback on the last generated SQL}")
    print("  /approved")
    print("  /list_commands")
    print("  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights.")
    print("\nEnjoy using the PostgreSQL Co-Pilot!")

if __name__ == "__main__":
    print("Starting pre-requisite setup for PostgreSQL Co-Pilot...")
    
    print("\nStep 1: Creating 'passwords' directory and template '.env' file...")
    if not create_passwords_dir_and_env_file():
        print("Failed to create password directory or .env file. Please check permissions and try manually.", file=sys.stderr)
        sys.exit(1)

    print("\nStep 2: Creating 'memory' directories...")
    if not create_memory_directories():
        print("Failed to create memory directories. Please check permissions and try manually.", file=sys.stderr)
        sys.exit(1)

    print("\nStep 3: Checking for pip and venv...")
    if not check_pip_and_venv():
        print("Critical Python components (pip/venv) missing. Please address this before proceeding.", file=sys.stderr)
    
    display_setup_instructions()
    print("\nPre-requisite setup script finished.")
    print("Please follow the instructions above to complete your setup and configure your .env file.")
