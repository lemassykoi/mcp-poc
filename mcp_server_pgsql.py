import os
import re
import json
import asyncpg
from contextlib import asynccontextmanager
from typing import AsyncIterator
import urllib.parse
from mcp.server.fastmcp import FastMCP, Context

# Global connection variable
conn_global = None

# POSTGRESQL INFOS
PG_USER = 'mcp_server'
PG_PASS = 'password'
PG_HOST = '127.0.0.1'
PG_PORT = 5432
PG_DB   = 'mcp_db'

DB_URI = f"postgres://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=disable"
PG_URI = f"postgres://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}"

# ------------------------------
# Lifespan: Connect to PostgreSQL
# ------------------------------
def update_db_in_url(url: str, new_db: str) -> str:
    """
    Update the database name in a PostgreSQL connection URL.
    """
    parsed = urllib.parse.urlparse(url)
    new_path = f"/{new_db}?sslmode=disable"
    new_url = parsed._replace(path=new_path)
    return urllib.parse.urlunparse(new_url)

# Global variables for state management
active_db   = None
db_pools    = {}
conn_global = None

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    global conn_global, active_db, db_pools
    # Set the default active database (e.g., "postgres")
    active_db = os.environ.get("DEFAULT_DB", PG_DB)
    # Get the base connection string (pointing to an administrative database)
    database_url = os.environ.get("DATABASE_URL", DB_URI)
    
    # Create a global connection for admin tasks if needed
    conn_global = await asyncpg.connect(database_url)
    # Create a connection pool for the default active database
    default_pool = await asyncpg.create_pool(dsn=database_url)
    db_pools[active_db] = default_pool

    try:
        # Yield the lifespan context containing state
        yield {"conn": conn_global, "active_db": active_db, "db_pools": db_pools}
    finally:
        await conn_global.close()
        # Close all connection pools
        for pool in db_pools.values():
            await pool.close()
        conn_global = None

# Create the MCP server instance with lifespan support.
mcp = FastMCP(
    "PostgreSQL Analytics Server",
    instructions="You are a SQL senior expert. You are in charge of a PostgreSQL instance. Don't explain or give SQL commands to user. When querying for tables, always exclude internal PG tables, unless explicitly asked.",
    lifespan=app_lifespan
)

# ---------------------------------------------------
# Resource: Expose table schemas from the PostgreSQL DB
# NOT USED
# ---------------------------------------------------
@mcp.resource("schemas://all")
async def get_schemas() -> dict:
    """
    Query the PostgreSQL information schema to retrieve table and column details.
    Returns a dictionary mapping table names to lists of columns (name and data type).
    """
    query = """
      SELECT table_name, column_name, data_type
      FROM information_schema.columns
      WHERE table_schema = 'public'
    """
    rows = await conn_global.fetch(query)
    schemas = {}
    for row in rows:
        table = row["table_name"]
        schemas.setdefault(table, []).append({
            "name": row["column_name"],
            "type": row["data_type"]
        })
    return schemas

# --------------------------------------------------------
# Tool: Execute read-only SQL queries against the PostgreSQL active DB
# --------------------------------------------------------
READ_ONLY_REGEX = re.compile(r'^\s*SELECT\b', re.IGNORECASE)

def is_read_only(query: str) -> bool:
    """Simple check to ensure the query starts with SELECT."""
    return READ_ONLY_REGEX.match(query) is not None

@mcp.tool()
async def run_select_query(query: str, ctx: Context) -> dict:
    """
    Executes a user-supplied SQL query on the active PostgreSQL database. Only queries that begin with SELECT (read-only) are allowed.
    
        Returns the results as a dictionary with a success flag and message.
    """
    if not is_read_only(query):
        return {"success": False, "error": "Only SELECT queries are allowed with this tool. To insert data, try 'insert_data'."}

    try:
        # Retrieve the active database and its connection pool from the lifespan context.
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]
        pool = lifespan["db_pools"][current_db]
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
            result = [dict(row) for row in rows]
        
        return json.dumps({"success": True, "message": result})
    except Exception as e:
        error_msg = str(e).encode("unicode_escape").decode("utf-8")
        return json.dumps({"success": False, "error": f"Error executing query: {error_msg}"})


# --------------------------------------------------------
# Tool: Create a new PostgreSQL database
# --------------------------------------------------------
VALID_DB_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

@mcp.tool()
async def create_database(db_name: str) -> dict:
    """
    Creates a new PostgreSQL database with the given name.
    The caller must ensure that the connected user has sufficient privileges.
    Returns a JSON response indicating success or error.
    """
    if not VALID_DB_NAME.match(db_name):
        print('invalid DB name')
        return json.dumps({"success": False, "error": "Invalid database name. Use only letters, numbers, and underscores."})
    
    conn = conn_global
    try:
        # PostgreSQL does not support parameterized names, so we ensure the db_name is safe
        query = f'CREATE DATABASE "{db_name}"'
        await conn.execute(query)
        return json.dumps({"success": True, "message": f"Database '{db_name}' created successfully."})
    except Exception as e:
        error_msg = str(e).encode("unicode_escape").decode("utf-8")
        return json.dumps({"success": False, "error": f"Error creating database: {error_msg}"})

# --------------------------------------------------------
# Tool: Create a new table in the active database
# --------------------------------------------------------
@mcp.tool()
async def create_table(table_name: str, columns: str, ctx: Context) -> dict:
    """
    Creates a new table with the given name and column definitions on the currently active database.
    
    Parameters:
        table_name (str): Name of the table (must contain only letters, numbers, or underscores)
        columns (str): A string with column definitions (e.g., "id SERIAL PRIMARY KEY, name VARCHAR(100)")
      
    Returns:
        A dictionary indicating success or failure.
    """
    if not VALID_DB_NAME.match(table_name):
        return json.dumps({"success": False, "error": "Invalid table name. Use only letters, numbers, and underscores."})
    
    if not columns.strip():
        return json.dumps({"success": False, "error": "Column definitions cannot be empty."})
    
    try:
        # Retrieve the active database and its connection pool from the lifespan context.
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]
        pool = lifespan["db_pools"][current_db]
        
        async with pool.acquire() as conn:
            # Build and execute the CREATE TABLE query.
            query = f'CREATE TABLE "{table_name}" ({columns})'
            await conn.execute(query)
        
        return json.dumps({"success": True, "message": f"Table '{table_name}' created successfully in database '{current_db}'."})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error creating table: {str(e)}"})


# --------------------------------------------------------
# Tool: Insert data into table in the active database
# --------------------------------------------------------
@mcp.tool()
async def insert_data(table_name: str, data: dict, ctx: Context) -> dict:
    """
    Inserts a new row into the specified table on the currently active database.
    
    Parameters:
      - table_name: Name of the table to insert data into.
      - data: A dictionary where keys are column names and values are the data to insert.
      
    Returns:
      A dictionary indicating success or error.
    """
    if not VALID_DB_NAME.match(table_name):
        return json.dumps({"success": False, "error": "Invalid table name. Use only letters, numbers, and underscores."})
    
    if not data:
        return json.dumps({"success": False, "error": "Data dictionary cannot be empty."})
    
    # Build column names and parameter placeholders
    columns = ", ".join(f'"{col}"' for col in data.keys())
    placeholders = ", ".join(f'${i}' for i in range(1, len(data) + 1))
    values = list(data.values())
    
    query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'
    
    try:
        # Retrieve the active database and its connection pool from the lifespan context.
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]
        pool = lifespan["db_pools"][current_db]
        
        async with pool.acquire() as conn:
            await conn.execute(query, *values)
        
        return json.dumps({"success": True, "message": f"Data inserted into '{table_name}' successfully."})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error inserting data: {str(e)}"})


@mcp.tool()
async def update_data(table_name: str, set_data: dict, where_condition: dict, ctx: Context) -> dict:
    """
    Updates rows in the specified table that meet a certain condition on the currently active database.

    Parameters:
      - table_name: Name of the table to update data into.
      - set_data: A dictionary where keys are column names and values are the updated data.
      - where_condition: A dictionary specifying the condition for which rows will be updated. The key is a column name, and the value is the required condition (e.g., {'id': 1} to update the row with id=1).

    Returns:
      A dictionary indicating success or error.
    """
    if not VALID_DB_NAME.match(table_name):
        return json.dumps({"success": False, "error": "Invalid table name. Use only letters, numbers, and underscores."})

    if not set_data:
        return json.dumps({"success": False, "error": "Data dictionary cannot be empty for updating data."})

    if not where_condition:
        return json.dumps({"success": False, "error": "Condition dictionary cannot be empty for updating data."})

    # Build SET clause
    set_clause = ", ".join(f'"{col}"=${i}' for i, col in enumerate(set_data.keys(), start=1))

    # Build WHERE clause
    where_clause = " AND ".join(f'"{key}"=$${i+len(set_data)}' for i, key in enumerate(where_condition.keys()))

    query = f'UPDATE "{table_name}" SET {set_clause} WHERE {where_clause}'

    # Combine parameters
    values = list(set_data.values()) + list(where_condition.values())

    try:
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]
        pool = lifespan["db_pools"][current_db]

        async with pool.acquire() as conn:
            await conn.execute(query, *values)

        return json.dumps({"success": True, "message": f"Data updated in '{table_name}' successfully."})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error updating data: {str(e)}"})


@mcp.tool()
async def alter_table(operation: str, table_name: str, params: dict, ctx: Context) -> dict:
    """
    Alters a table's structure in the active database using a unified interface.
    
    Parameters:
      - operation: The type of alter operation ("add", "rename", "drop").
      - table_name: Name of the table (must use only letters, numbers, and underscores).
      - params: A dictionary with additional parameters:
          For "add": expects "column_definition" (e.g., "age INT")
          For "rename": expects "old_column" and "new_column"
          For "drop": expects "column" (the name of the column to drop)
    
    Returns:
      A dictionary indicating success or error.
    """
    if not VALID_DB_NAME.match(table_name):
        return json.dumps({"success": False, "error": "Invalid table name."})
    
    query = ""
    if operation == "add":
        column_definition = params.get("column_definition")
        if not column_definition:
            return json.dumps({"success": False, "error": "Missing 'column_definition' for add operation."})
        query = f'ALTER TABLE "{table_name}" ADD COLUMN {column_definition}'
    elif operation == "rename":
        old_column = params.get("old_column")
        new_column = params.get("new_column")
        if not old_column or not new_column:
            return json.dumps({"success": False, "error": "Missing 'old_column' or 'new_column' for rename operation."})
        if not VALID_DB_NAME.match(old_column) or not VALID_DB_NAME.match(new_column):
            return json.dumps({"success": False, "error": "Invalid column names."})
        query = f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_column}" TO "{new_column}"'
    elif operation == "drop":
        column = params.get("column")
        if not column:
            return json.dumps({"success": False, "error": "Missing 'column' for drop operation."})
        if not VALID_DB_NAME.match(column):
            return json.dumps({"success": False, "error": "Invalid column name."})
        query = f'ALTER TABLE "{table_name}" DROP COLUMN "{column}"'
    else:
        return json.dumps({"success": False, "error": f"Unsupported operation '{operation}'. Use 'add', 'rename', or 'drop'."})
    
    try:
        # Retrieve the active database's connection pool from the lifespan context.
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]
        pool = lifespan["db_pools"][current_db]
        
        async with pool.acquire() as conn:
            await conn.execute(query)
        
        return json.dumps({"success": True, "message": f"Table '{table_name}' altered successfully: {query}"})

    except Exception as e:
        return json.dumps({'success': False, 'error': f'Error altering table: {str(e)}'})


# --------------------------------------------------------
# Tool: List all databases
# --------------------------------------------------------
@mcp.tool()
async def list_databases(ctx: Context) -> dict:
    """
    Lists all available databases in the PostgreSQL instance using the currently active database's connection pool.

    Returns:
      A dictionary with the list of database names.
    """
    try:
        # Retrieve lifespan context which includes our active_db and db_pools
        lifespan = ctx.request_context.lifespan_context
        current_db = lifespan["active_db"]  # active_db stored during lifespan initialization/switch
        pool = lifespan["db_pools"][current_db]
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = [row["datname"] for row in rows]
        return json.dumps({"success": True, "message": databases})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error fetching databases: {str(e)}"})



# --------------------------------------------------------
# Tool: Switch active DB
# --------------------------------------------------------
@mcp.tool()
async def switch_db(target_db: str, ctx: Context) -> dict:
    """
    Switches the active database in the lifespan context to the specified target database.
    If a connection pool for the target database does not exist, it creates one.
    
    Parameters:
      - target_db: The name of the target database to switch to.
      
    Returns:
      A dictionary indicating success or error.
    """
    lifespan = ctx.request_context.lifespan_context
    lifespan["active_db"] = target_db
    db_pools = lifespan["db_pools"]
    
    if target_db not in db_pools:
        base_db_url = os.environ.get("POSTGRESQL_URL", PG_URI)
        target_db_url = update_db_in_url(base_db_url, target_db)
        try:
            db_pools[target_db] = await asyncpg.create_pool(dsn=target_db_url)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    return json.dumps({"success": True, "message": f"Switched active database to '{target_db}'."})



# --------------------------------------------------------
# Tool: Get current active DB
# --------------------------------------------------------
@mcp.tool()
def get_current_active_db(ctx: Context) -> dict:
    """
    Returns the currently active database from the lifespan context.
    """
    lifespan = ctx.request_context.lifespan_context
    return json.dumps({"success": True, "message": lifespan["active_db"]})



# --------------------------------------------------------
# Tool: Get Server Infos
# --------------------------------------------------------
@mcp.tool()
async def get_server_info(ctx: Context) -> dict:
    """
    Retrieves information about the connected PostgreSQL server.
    
    Returns:
      A dictionary with details such as version, uptime, and system info.
    """
    try:
        conn = ctx.request_context.lifespan_context["conn"]
        row = await conn.fetchrow("""
            SELECT 
                version() AS pg_version, 
                current_database() AS current_db, 
                inet_server_addr() AS server_ip, 
                inet_server_port() AS server_port, 
                pg_postmaster_start_time() AS start_time
        """)
        return {
            "success": True,
            "message": {
                "PostgreSQL Version": str(row["pg_version"]),
                "Current Database": str(row["current_db"]),
                "Server IP": str(row["server_ip"]),
                "Server Port": str(row["server_port"]),
                "Uptime Since": str(row["start_time"])
            }
        }
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error fetching server info: {str(e)}"})


# ---------------------------------------------------
# Prompts: Provide example SQL prompts for data analysis
# ---------------------------------------------------
@mcp.prompt()
def count_rows_prompt(table: str) -> str:
    """Return a prompt to count the number of rows in a given table."""
    return f"SELECT COUNT(*) FROM {table};"

@mcp.prompt()
def top_records_prompt(table: str, limit: int = 10) -> str:
    """Return a prompt to retrieve the top N records from a table."""
    return f"SELECT * FROM {table} LIMIT {limit};"

@mcp.prompt()
def group_by_prompt(table: str, column: str) -> str:
    """Return a prompt to group table data by a column and count the records."""
    return f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column};"

# -----------------------------------
# Run the MCP server if executed directly.
# -----------------------------------
if __name__ == '__main__':
    mcp.run()