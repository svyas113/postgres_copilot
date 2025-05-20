import sys
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# Import necessary components from the mcp SDK
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filename='postgres_server.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
mcp_logger = logging.getLogger('mcp')
mcp_logger.setLevel(logging.DEBUG)
if not mcp_logger.handlers:
    file_handler = logging.FileHandler('postgres_server.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    mcp_logger.addHandler(file_handler)

logger.info("PostgreSQL MCP Server script started.")

# Global variable to hold the database connection
db_connection = None

# Create a FastMCP server instance
logger.info("Creating FastMCP instance for PostgreSQL server...")
mcp = FastMCP("PostgreSQLServer")
logger.info("FastMCP instance created.")

@mcp.tool()
def connect_to_postgres(connection_string: str) -> str:
    """
    Connects to a PostgreSQL database using the provided connection string.
    Example connection string: "postgresql://user:password@host:port/dbname"
    """
    global db_connection
    logger.debug(f"Tool 'connect_to_postgres' called with connection_string: {connection_string}")
    try:
        if db_connection:
            db_connection.close()
            logger.info("Closed existing database connection.")
        db_connection = psycopg2.connect(connection_string)
        logger.info("Successfully connected to PostgreSQL database.")
        return "Successfully connected to PostgreSQL database."
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
        db_connection = None
        return f"Error: Failed to connect to PostgreSQL. {e}"

@mcp.tool()
def get_schema_and_sample_data() -> str | dict:
    """
    Fetches CREATE TABLE-like statements (column definitions) and 5 sample rows
    for each table in the currently connected PostgreSQL database.
    Requires a prior successful connection using 'connect_to_postgres'.
    """
    global db_connection
    logger.debug("Tool 'get_schema_and_sample_data' called.")
    if not db_connection:
        logger.warning("No active database connection for 'get_schema_and_sample_data'.")
        return "Error: Not connected to any database. Please use 'connect_to_postgres' first."

    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all user tables (excluding system tables)
            cur.execute("""
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
            """)
            tables = [row['tablename'] for row in cur.fetchall()]
            logger.debug(f"Found tables: {tables}")

            result = {}
            for table_name in tables:
                table_info = {"schema": "", "sample_data": []}

                # Get column definitions (simplified CREATE TABLE)
                cur.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (table_name,))
                columns = cur.fetchall()

                create_statement_parts = [f"CREATE TABLE {table_name} ("]
                for col in columns:
                    part = f"  {col['column_name']} {col['data_type']}"
                    if col['is_nullable'] == 'NO':
                        part += " NOT NULL"
                    if col['column_default'] is not None:
                        part += f" DEFAULT {col['column_default']}"
                    create_statement_parts.append(part + ",")
                if create_statement_parts[-1].endswith(","): # remove last comma
                     create_statement_parts[-1] = create_statement_parts[-1][:-1]
                create_statement_parts.append(");")
                table_info["schema"] = "\n".join(create_statement_parts)

                # Get 5 sample rows
                # Using sql.Identifier for safe table name interpolation
                query = sql.SQL("SELECT * FROM {} LIMIT 5;").format(sql.Identifier(table_name))
                cur.execute(query)
                sample_data = cur.fetchall()
                table_info["sample_data"] = sample_data

                result[table_name] = table_info
            
            logger.info(f"Successfully fetched schema and sample data for {len(tables)} tables.")
            return result if result else "No user tables found or database is empty."

    except Exception as e:
        logger.error(f"Error fetching schema and data: {e}", exc_info=True)
        return f"Error: Could not fetch schema and data. {e}"

@mcp.tool()
def execute_postgres_query(query: str) -> str | list:
    """
    Executes a given SQL query on the currently connected PostgreSQL database.
    Requires a prior successful connection using 'connect_to_postgres'.
    For SELECT queries, it returns a list of rows (as dictionaries).
    For other queries (INSERT, UPDATE, DELETE), it returns a success message with row count.
    """
    global db_connection
    logger.debug(f"Tool 'execute_postgres_query' called with query: {query[:100]}...") # Log truncated query
    if not db_connection:
        logger.warning("No active database connection for 'execute_postgres_query'.")
        return "Error: Not connected to any database. Please use 'connect_to_postgres' first."

    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            
            # For SELECT queries, fetch results
            if cur.description: # Check if the cursor has a description (i.e., it's a SELECT-like query)
                results = cur.fetchall()
                logger.info(f"Query executed successfully. Fetched {len(results)} rows.")
                db_connection.commit() # Commit any implicit transaction if any (though SELECTs usually don't need it)
                return results
            else:
                # For INSERT, UPDATE, DELETE, etc.
                rowcount = cur.rowcount
                db_connection.commit() # Important to commit changes
                logger.info(f"Query executed successfully. {rowcount} rows affected.")
                return f"Query executed successfully. {rowcount} rows affected."
                
    except Exception as e:
        db_connection.rollback() # Rollback in case of error
        logger.error(f"Error executing query: {e}", exc_info=True)
        return f"Error: Could not execute query. {e}"

logger.info("PostgreSQL MCP tools defined.")

if __name__ == "__main__":
    logger.info("__main__ block started for PostgreSQL server.")
    if sys.platform == "win32":
        logger.info("Configuring stdout/stdin for win32.")
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
        logger.info("stdout/stdin configured for win32.")

    logger.info("Starting MCP PostgreSQL Server (stdio)...")
    print("Starting MCP PostgreSQL Server (stdio)...")

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.critical(f"Critical error running MCP PostgreSQL server: {e}", exc_info=True)
    finally:
        if db_connection:
            db_connection.close()
            logger.info("Closed database connection on server exit.")
        logger.info("MCP PostgreSQL server finished or exited.")
        logging.shutdown()