import json
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List, Union


class PostgreSQLSession:
    """
    A session handler for PostgreSQL database connections.
    Provides methods for executing SQL queries and other database operations.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the PostgreSQL session with a connection string.
        
        Args:
            connection_string: PostgreSQL connection string in format 
                              "postgresql://user:password@host:port/dbname"
        """
        self.connection_string = connection_string
        self.connection = None
        self.connect()
    
    def connect(self) -> bool:
        """
        Establish a connection to the PostgreSQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.connection:
                self.connection.close()
                print("[INFO] Closed existing database connection.")
            
            self.connection = psycopg2.connect(self.connection_string)
            print("[INFO] Successfully connected to PostgreSQL database.")
            
            # Test the connection with a simple query
            with self.connection.cursor() as cur:
                cur.execute("SELECT 1")
            
            return True
        except Exception as e:
            error_msg = f"Failed to connect to PostgreSQL: {e}"
            print(f"[ERROR] {error_msg}")
            self.connection = None
            return False
    
    def is_connected(self) -> bool:
        """
        Check if the session is connected to the database.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            # Try a simple query to check connection
            with self.connection.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the session."""
        return f"PostgreSQLSession(connected={self.is_connected()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the session."""
        return f"PostgreSQLSession(connection_string='{self.connection_string}', connected={self.is_connected()})"
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database tool by name with the given parameters.
        
        Args:
            tool_name: The name of the tool to call
            params: Parameters for the tool
            
        Returns:
            The result of the tool execution
        """
        if tool_name == "execute_postgres_query":
            return await self.execute_query(params.get("query", ""), params.get("row_limit", 100))
        elif tool_name == "describe_table":
            return await self.describe_table(params.get("table_name", ""))
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    async def execute_query(self, query: str, row_limit: int = 100) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results.
        
        Args:
            query: The SQL query to execute
            row_limit: Maximum number of rows to return
            
        Returns:
            Dictionary with status and data/message
        """
        if not self.is_connected():
            if not self.connect():
                return {"status": "error", "message": "Not connected to database"}
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                
                # For SELECT queries, fetch results
                if query.strip().upper().startswith("SELECT"):
                    rows = cur.fetchall()
                    # Convert rows to list of dicts
                    result = [dict(row) for row in rows]
                    
                    # Apply row limit
                    if row_limit > 0:
                        result = result[:row_limit]
                    
                    return {
                        "status": "success",
                        "data": result,
                        "row_count": len(result),
                        "total_rows": cur.rowcount
                    }
                # For other queries (INSERT, UPDATE, DELETE), return affected rows
                else:
                    self.connection.commit()
                    return {
                        "status": "success",
                        "affected_rows": cur.rowcount,
                        "message": f"Query executed successfully. {cur.rowcount} rows affected."
                    }
        except Exception as e:
            # Rollback in case of error
            if self.connection:
                self.connection.rollback()
            
            error_message = str(e)
            return {
                "status": "error",
                "message": error_message
            }
    
    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a table.
        
        Args:
            table_name: The name of the table to describe
            
        Returns:
            Dictionary with table information
        """
        if not self.is_connected():
            if not self.connect():
                return {"status": "error", "message": "Not connected to database"}
        
        try:
            result = {
                "status": "success",
                "table_name": table_name,
                "columns": [],
                "primary_key": [],
                "foreign_keys": [],
                "sample_data": []
            }
            
            # Extract schema and table name
            parts = table_name.split(".")
            if len(parts) > 1:
                schema_name = parts[0]
                table_name_only = parts[1]
            else:
                schema_name = "public"
                table_name_only = table_name
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                # Get column information
                cur.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema_name, table_name_only))
                
                columns = cur.fetchall()
                result["columns"] = [dict(col) for col in columns]
                
                # Get primary key information
                cur.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                """, (schema_name, table_name_only))
                
                pk_columns = cur.fetchall()
                result["primary_key"] = [col["column_name"] for col in pk_columns]
                
                # Get foreign key information
                cur.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                """, (schema_name, table_name_only))
                
                fk_info = cur.fetchall()
                for fk in fk_info:
                    result["foreign_keys"].append({
                        "column": fk["column_name"],
                        "reference": f"{fk['foreign_table_schema']}.{fk['foreign_table_name']}.{fk['foreign_column_name']}"
                    })
                
                # Get sample data (5 rows)
                try:
                    query = sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name_only)
                    )
                    cur.execute(query)
                    sample_data = cur.fetchall()
                    result["sample_data"] = [dict(row) for row in sample_data]
                except Exception as e:
                    result["sample_data_error"] = str(e)
            
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("[INFO] Database connection closed.")
