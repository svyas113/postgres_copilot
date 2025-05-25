# PostgreSQL Co-Pilot Chat: Technical Architecture

This document outlines the high-level technical architecture of the `postgres_copilot_chat.py` application, a command-line interface (CLI) tool designed to assist users in interacting with PostgreSQL databases using natural language queries and commands. It leverages a Large Language Model (LLM) via LiteLLM for understanding user intent and generating SQL, and communicates with a Model Context Protocol (MCP) server for database operations.

## Core Components

The system is composed of several key Python modules and a central client application:

1.  **`postgres_copilot_chat.py` (Main Client Application)**:
    *   **Role**: Orchestrates the entire user interaction flow. It's the primary entry point and acts as the MCP client.
    *   **Responsibilities**:
        *   Manages the main chat loop (agent loop), accepting user input.
        *   Parses user input for commands or natural language queries.
        *   Initializes and manages the connection to the MCP server.
        *   Initializes and manages the LiteLLM client (`LiteLLMMcpClient`).
        *   Dispatches tasks to specialized modules.
        *   Maintains application state (DB connection, schema, feedback cycles).
        *   Handles resource cleanup.
    *   **Key Class**: `LiteLLMMcpClient`
        *   Encapsulates LLM interaction logic.
        *   Manages the MCP session.
        *   Stores application state.

2.  **`postgresql_server.py` (MCP Server - External Process)**:
    *   **Role**: Bridge between the client and the PostgreSQL database.
    *   **Responsibilities**:
        *   Exposes MCP tools: `connect_to_postgres`, `get_schema_and_sample_data`, `execute_postgres_query`.
        *   Manages database connections (e.g., using `psycopg2`).
        *   Serializes database results/schema to the client.
    *   **Note**: Runs as a separate process.

3.  **Specialized Modules (within `db-copilot/`)**:
    *   **`initialization_module.py`**: Handles new database connection setup (connect, fetch schema, save schema via `memory_module`).
    *   **`sql_generation_module.py`**: Generates SQL from natural language using LLM, validates, and verifies SQL via MCP server.
    *   **`insights_module.py`**: Extracts and manages persistent insights from feedback reports using LLM and `memory_module`.
    *   **`database_navigation_module.py`**: Handles general NL queries about the database, deciding actions (execute SQL, get schema, answer directly, clarify) via LLM. If schema is fetched, a second LLM call synthesizes an answer.
    *   **`memory_module.py`**: Manages persistence of schemas, feedback reports, insights, and conversation history to the file system.
    *   **`pydantic_models.py`**: Defines Pydantic models for data structures and validation (LLM responses, feedback reports, insights).

## Model Usage and LLM Interaction

*   **LLM Abstraction**: LiteLLM is used as an abstraction layer, allowing flexibility in choosing the underlying LLM provider (e.g., Gemini, OpenAI models). The current configuration defaults to a Gemini model (`gemini/gemini-2.5-pro-preview-05-06`).
*   **API Key Management**: API keys for the LLM provider are loaded from a `.env` file located in `db-copilot/passwords/`. LiteLLM uses environment variables like `GEMINI_API_KEY` or `OPENAI_API_KEY`.
*   **Prompt Engineering**: Each module requiring LLM interaction (`sql_generation_module`, `insights_module`, `database_navigation_module`) is responsible for constructing specific, detailed prompts tailored to its task. These prompts include:
    *   System instructions defining the AI's role and output requirements (e.g., JSON format).
    *   User's query or relevant data (e.g., feedback report content).
    *   Contextual information like database schema, sample data, and cumulative insights.
    *   Pydantic model schemas to guide the LLM in generating structured JSON output.
*   **Tool Integration (with LLM)**: The `LiteLLMMcpClient` converts MCP tools provided by `postgresql_server.py` into a format compatible with the LLM's tool-calling capabilities (OpenAI-compatible format). The LLM can then request the execution of these tools. The client handles the execution via the MCP session and sends the results back to the LLM.
*   **Response Processing**: LLM responses, especially those expected in JSON format, are validated against Pydantic models. Retry mechanisms are implemented in some modules (e.g., `sql_generation_module`) to handle malformed or incorrect LLM outputs by re-prompting with error feedback.
*   **Conversation History**: `LiteLLMMcpClient` maintains a list of messages (user, assistant, system, tool) which forms the conversation history. This history is passed to the LLM for subsequent calls to maintain context.

## Agent Loop (Chat Loop in `postgres_copilot_chat.py`)

The `chat_loop` method within `LiteLLMMcpClient` drives the user interaction:

1.  **Initialization**:
    *   The client connects to the MCP server (`postgresql_server.py`).
    *   MCP tools are listed and prepared for LiteLLM.
    *   A system prompt is added to the conversation history.
2.  **User Input**: The loop waits for user input from the command line.
3.  **Command Dispatch (`dispatch_command`)**:
    *   Input is parsed.
    *   **Connection Handling**: If a connection string or `/change_database` command is given, `_handle_initialization` is called, which uses `initialization_module.py` to connect to the DB, fetch schema, and save it.
    *   **SQL Generation (`/generate_sql`)**: `sql_generation_module.generate_sql_query()` is invoked. This involves LLM calls for SQL generation and MCP tool calls for execution.
    *   **Feedback (`/feedback`)**: The provided feedback text is used to construct a prompt for the LLM to refine the previous SQL query and update the `FeedbackReportContentModel`. This involves an LLM call expecting a JSON update.
    *   **Approval (`/approved`)**: The current `FeedbackReportContentModel` is saved via `memory_module`. Then, `insights_module.generate_and_update_insights()` is called, which involves an LLM call to process the feedback report and update persistent insights.
    *   **Navigation (Natural Language Query)**: If the input is not a command, `database_navigation_module.handle_navigation_query()` is called. This involves an LLM call to decide an action, potentially followed by MCP tool calls and/or further LLM calls to synthesize an answer.
    *   **Other Commands**: `/list_commands`.
4.  **Response to User**: The result from the dispatched command/module is printed to the console.
5.  **Loop**: The process repeats from step 2.
6.  **Cleanup**: On 'quit' or program termination, `cleanup()` is called to close the MCP session and perform any other necessary cleanup. Conversation history is saved by `memory_module`.

## Data Flow

1.  **User Input**: CLI -> `postgres_copilot_chat.py` (`LiteLLMMcpClient`).
2.  **Contextual Data for LLM**:
    *   Schema/Sample Data: PostgreSQL DB -> MCP Server (`get_schema_and_sample_data` tool) -> `initialization_module` -> `memory_module` (save) -> `LiteLLMMcpClient` (load for prompts).
    *   Insights: `memory_module` (load) -> `LiteLLMMcpClient` (load for prompts).
    *   Conversation History: Maintained by `LiteLLMMcpClient`.
3.  **LLM Interaction**:
    *   `postgres_copilot_chat.py` (via specialized modules) -> `LiteLLMMcpClient._send_message_to_llm()` -> LiteLLM -> LLM Provider API.
    *   LLM Provider API -> LiteLLM -> `LiteLLMMcpClient._process_llm_response()` -> Specialized module.
4.  **Tool Execution (Requested by LLM or directly by modules)**:
    *   `LiteLLMMcpClient` / Specialized Module -> `client.session.call_tool()` -> MCP Server (`postgresql_server.py`).
    *   MCP Server -> PostgreSQL DB.
    *   PostgreSQL DB -> MCP Server -> `LiteLLMMcpClient` / Specialized Module.
5.  **Persistent Storage (`memory_module`)**:
    *   Schema, Feedback Reports, Insights, Conversation History <-> File System (`db-copilot/memory/`).
6.  **Output to User**: Specialized module -> `postgres_copilot_chat.py` -> CLI.

**Diagrammatic Flow (Simplified for SQL Generation):**

```
User Input --> [postgres_copilot_chat.py: LiteLLMMcpClient]
              |        ^
              |        | (NLQ, Schema, Insights)
              v        |
      [sql_generation_module] ----> [LiteLLM API] ----> LLM (e.g., Gemini)
              |        ^              (Prompt)         (JSON: SQL, Explanation)
              |        | (Generated SQL)
              v        |
      [MCP Server: postgresql_server.py] -- (SQL Query) --> [PostgreSQL DB]
              |        ^
              |        | (Query Results)
              <--------
              |
              v
Output to User
```

## Infrastructure & Dependencies

*   **Runtime Environment**: Python 3.x.
*   **Key Libraries**:
    *   `litellm`: For interacting with various LLM APIs.
    *   `mcp.py`: For Model Context Protocol client-server communication.
    *   `python-dotenv`: For loading environment variables (API keys).
    *   `pydantic`: For data validation and settings management.
    *   `asyncio`: For asynchronous operations.
*   **External Services**:
    *   An LLM provider (e.g., Google for Gemini models) accessible via API.
    *   A running PostgreSQL database instance.
*   **Local Setup**:
    *   The `postgres_copilot_chat.py` script is run from the command line.
    *   It launches `postgresql_server.py` as a subprocess, which handles direct database connections.
    *   A `db-copilot/passwords/.env` file is required to store the LLM API key.
    *   The `db-copilot/memory/` directory is used for persistent storage of schemas, insights, feedback, and conversation history.

## Key Design Principles

*   **Modularity**: Functionality is broken down into specialized modules.
*   **LLM-Driven Intelligence**: Core tasks like SQL generation, feedback processing, and navigation decisions are powered by an LLM.
*   **MCP for Database Interaction**: Decouples the main application from direct database driver management, allowing the MCP server to handle database specifics.
*   **State Management**: The `LiteLLMMcpClient` class and `memory_module` are crucial for maintaining session state and persisting data.
*   **Pydantic for Data Validation**: Ensures structured and validated data exchange, especially with LLM outputs.
*   **User-Centric Interaction**: Commands and natural language processing aim for a user-friendly experience.
*   **Iterative Refinement**: The feedback loop (`/feedback`, `/approved`) allows for iterative improvement of SQL queries and knowledge capture.

This architecture allows for a flexible and extensible system where different LLMs can be swapped out (thanks to LiteLLM) and database interaction logic is encapsulated within the MCP server.
