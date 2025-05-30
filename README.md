# PostgreSQL Co-Pilot

PostgreSQL Co-Pilot is an intelligent assistant designed to help users interact with PostgreSQL databases. It leverages Google's Gemini generative AI models to understand natural language queries, generate SQL, process user feedback, and extract valuable insights about database schemas and query patterns. The Co-Pilot aims to streamline database interaction, making it more intuitive and efficient, especially for users who may not be SQL experts.

## Features

*   **Natural Language to SQL:** Converts user questions in plain English into executable PostgreSQL queries.
*   **Interactive Feedback Loop:** Allows users to provide feedback on generated SQL, which the Co-Pilot uses to refine queries and improve its understanding.
*   **Automated Insights Generation:** Learns from user interactions and feedback to build a cumulative knowledge base (`summarized_insights.md`) about database schemas, common query patterns, and SQL best practices.
*   **Database Schema Caching:** Stores database schema and sample data locally for faster subsequent interactions and offline reference.
*   **Conversation History:** Saves chat logs for review and context.
*   **Modular Design:** Separates concerns into distinct Python modules for initialization, SQL generation, insights management, database navigation, and memory operations.
*   **MCP Integration:** Utilizes the Model Context Protocol (MCP) to interact with a local PostgreSQL server, enabling database operations like query execution and schema fetching.

## Directory Structure

The project uses the following directory structure:

```
memory/
├── conversation_history/
├── feedback/
├── insights/
└── NL2SQL/
```
(Note: The `passwords/` directory exists and is created by `prerequisites.py`. Inside the `.env` file located in the `passwords/` directory, the user has to enter their model ID and credentials to use that particular model. We use litellm now, so you can use any model that is compatible with litellm, but for now it is locked for AWS Bedrock, Claude, OpenAI, Google Generative AI, and Azure Cloud.)

## Modules

*   **`postgres_copilot_chat.py`**: The main entry point and chat interface for interacting with the Co-Pilot. Manages the overall workflow, user commands, and communication with the Gemini model and MCP server.
*   **`memory_module.py`**: Handles all file and directory operations for storing and retrieving feedback, insights, schema data, and conversation history.
*   **`initialization_module.py`**: Manages the initial connection to a PostgreSQL database, fetches schema and sample data, and saves it.
*   **`sql_generation_module.py`**: Responsible for taking a natural language question, database schema, and existing insights to generate an initial SQL query using the Gemini model.
*   **`insights_module.py`**: Processes approved feedback reports to extract and update cumulative insights in `summarized_insights.md`.
*   **`database_navigation_module.py`**: Allows users to ask questions about the connected database structure using the cached schema and insights.
*   **`pydantic_models.py`**: Defines Pydantic models for structured data handling, such as feedback reports and insights.
*   **`postgresql_server.py`**: (Assumed) The MCP server script that directly interacts with the PostgreSQL database to execute queries, fetch schema, etc. This is passed as an argument to `postgres_copilot_chat.py`.
*   **`.env`**: Stores environment variables, primarily the `GOOGLE_API_KEY`, located in the `db-copilot/` root directory.
*   **`pyproject.toml` & `uv.lock`**: Define project dependencies and manage the Python environment (likely using `uv` or a similar tool).

## Workflow

1.  **Initialization:**
    *   The user provides a PostgreSQL connection string (e.g., `postgresql://user:pass@host:port/dbname`) or uses the `/change_database` command.
    *   The `initialization_module` connects to the database, fetches its schema and some sample data.
    *   This information is saved locally in the `memory/schema/` directory by `memory_module.py`.
    *   Existing cumulative insights from `memory/insights/summarized_insights.md` are loaded.

2.  **SQL Generation:**
    *   The user asks a natural language question using the `/generate_sql` command (e.g., `/generate_sql: Show me all customers from California`).
    *   `sql_generation_module.py` uses the Gemini model, along with the cached schema and insights, to generate a SQL query.
    *   The generated SQL and an explanation are presented to the user.

3.  **Feedback (Optional):**
    *   If the SQL query is not perfect, the user can provide feedback using the `/feedback` command (e.g., `/feedback: The query should also filter by active status`).
    *   The Co-Pilot uses this feedback to generate a corrected SQL query and explanation. This can be an iterative process.

4.  **Approval & Insights Update:**
    *   Once the user is satisfied with the SQL query, they use the `/approved` command.
    *   `memory_module.py` saves a detailed feedback report (including the natural language question, initial SQL, iterations, final SQL, and LLM analysis) as a Markdown file in `memory/feedback/`.
    *   `insights_module.py` processes this feedback report. It uses the Gemini model to extract new insights or reinforce existing ones.
    *   These insights are then merged into the `memory/insights/summarized_insights.md` file.

5.  **Database Navigation:**
    *   Users can ask general questions about the database structure or data (without a `/` prefix).
    *   `database_navigation_module.py` attempts to answer these using the cached schema and cumulative insights.

## Setup and Installation

1.  **Clone the repository.**
2.  **Create a virtual environment and install dependencies:**
    *   It's recommended to use `uv` (if `uv.lock` and `pyproject.toml` are configured for it) or `pip`.
    *   Example with pip (refer to `pyproject.toml` for actual dependencies):
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        pip install google-generativeai python-dotenv pydantic mcp # Add other specific dependencies
        ```
3.  **Set up Environment Variables:**
    *   Ensure a `.env` file exists in the `db-copilot/` root directory.
    *   The application looks for `GOOGLE_API_KEY` in this file (`db-copilot/.env`).
    *   Add your Google API key to `db-copilot/.env`:
        ```
        # In db-copilot/.env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
    *   **Important:** Ensure you replace `"YOUR_GOOGLE_API_KEY_HERE"` with your actual Google Gemini API key. The file `db-copilot/passwords/.env` is not used by `postgres_copilot_chat.py` for loading this specific key.
4.  **Ensure PostgreSQL Server is Running:** The Co-Pilot connects to an existing PostgreSQL database.
5.  **MCP Server:** Make sure the `postgresql_server.py` (or equivalent MCP server script for PostgreSQL interaction) is available and executable.

To install and run the system:

Clone the repo,

If you have uv installed which is a python dependency manager you just have to run the uv sync command if you don't you have to install it and run uv sync (That will create a python virtual environment and install all the dependencies in there)

Once done just run prerequisites.py that will install all the necessary folders to run the postgres_copilot_chat.py which is the main chat bot and once done

Run python postgres_copilot_chat.py postgresql_server.py

That will start the chat bot

## Usage

Run the main chat interface from the `db-copilot` directory:

```bash
python postgres_copilot_chat.py <path_to_mcp_server_script.py>
```

Example:

```bash
python postgres_copilot_chat.py ./postgresql_server.py
```

Once started, you can interact with the Co-Pilot using the following commands:

*   **Initial Connection:** Simply type or paste your PostgreSQL connection string:
    `postgresql://user:password@host:port/dbname`
*   **Change Database:**
    `/change_database: "your_new_connection_string"`
*   **Generate SQL:**
    `/generate_sql: Your natural language question about the database`
    (e.g., `/generate_sql: List all employees hired in the last year`)
*   **Provide Feedback:**
    `/feedback: Your comments or corrections for the last generated SQL`
    (e.g., `/feedback: The date format for hiring is YYYY-MM-DD, not MM/DD/YYYY`)
*   **Approve SQL:**
    `/approved`
    (Saves the feedback report and updates insights)
*   **Navigate/Ask Questions:**
    Type your question directly without a command prefix to ask about the database schema or general information based on loaded insights.
    (e.g., `What tables are related to products?`)
*   **Quit:**
    `quit`

## Dependencies

*   `google-generativeai`: For interacting with Google's Gemini AI models.
*   `python-dotenv`: For managing environment variables (like API keys).
*   `pydantic`: For data validation and settings management using Python type annotations.
*   `mcp` (Model Context Protocol): For communication between the Co-Pilot client and the database interaction server.
*   (Other dependencies as listed in `pyproject.toml`)

## Future Enhancements

*   More sophisticated parsing of existing `summarized_insights.md` for merging new insights.
*   Support for other database systems beyond PostgreSQL.
*   Enhanced error handling and user guidance.
*   UI for easier interaction (e.g., web interface).
