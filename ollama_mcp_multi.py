import asyncio
import requests
import json
import urllib.parse
import colorama
import os
import sys
from typing import List, Optional, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Resource, Tool

B_RED        = colorama.Back.RED
RED          = colorama.Fore.RED
BLUE         = colorama.Fore.BLUE
CYAN         = colorama.Fore.CYAN
GREEN        = colorama.Fore.GREEN
YELLOW       = colorama.Fore.YELLOW
MAGENTA      = colorama.Fore.MAGENTA
YELLOW_LIGHT = colorama.Fore.LIGHTYELLOW_EX
RESET        = colorama.Style.RESET_ALL

# Global settings for Ollama API calls
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_SEED: int = 1234567890

# Initialize logger
logging.basicConfig(
    format = '%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-40s \t (%(filename)s:%(lineno)d)',
    stream = sys.stdout,
    level  = level
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Helper function to build a resource list (from MCP resources)
def build_resource_list(resources: List[Resource]) -> List[Dict[str, str]]:
    resource_list = []
    for resource in resources:
        if resource[0] == 'resources':
            for item in resource[1]:
                logger.debug(item)
                resource_dict = {
                    'name': item.name,
                    'uri': str(item.uri)  # Convert AnyUrl to string
                }
                resource_list.append(resource_dict)
    return resource_list

class MultiServerMCPClient:
    """
    A multi-server MCP client that supports connecting to multiple MCP servers,
    merging their tool lists (with namespaced tool names), and dispatching tool calls
    to the appropriate server.
    """
    def __init__(self):
        # Dictionary to hold MCP sessions keyed by server identifier.
        self.sessions: Dict[str, ClientSession] = {}
        # Exit stack for proper asynchronous cleanup.
        self.exit_stack = AsyncExitStack()
        self.ollama_api_url = OLLAMA_BASE_URL
        # Dictionaries to store each server's tools and resources.
        self.tools: Dict[str, List[Tool]] = {}
        self.resources: Dict[str, List[Dict[str, str]]] = {}
        logger.info("MultiServerMCPClient initialized.")

    def convert_mcp_tools_to_ollama_tools(self, tools: List[Tool], server_name: str) -> List[Dict]:
        """
        Convert MCP tools to a list of functions formatted for Ollama.
        Each tool name is prefixed with the server name to namespace the tool.
        """
        logger.info('Convert MCP Tools => Ollama Tools')
        result = []
        for tool in tools:
            function_dict = {
                "type": "function",
                "function": {
                    # Namespace the tool name with server identifier.
                    "name": f"{server_name}:{tool.name}",
                    "description": tool.description.strip(),
                    "parameters": tool.inputSchema
                }
            }
            result.append(function_dict)
        return result

    async def connect_to_server(self, server_name: str, server_script_path: str) -> None:
        """
        Connect to an MCP server using the provided script (either .py or .js),
        then retrieve and store its tools and resources.
        """
        # Determine the command based on file extension.
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        # Start the server process and get a stdio transport.
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        # Create and initialize a ClientSession.
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        self.sessions[server_name] = session

        # Retrieve resources and tools from the server.
        resources_raw = await session.list_resources()
        tools_raw = await session.list_tools()
        self.tools[server_name] = tools_raw.tools
        self.resources[server_name] = build_resource_list(resources_raw)
        print(f"\nConnected to server '{server_name}' with tools:",
              [tool.name for tool in tools_raw.tools])
        print('\n')

    async def call_tool(self, server_name: str, tool_name: str, arguments: str):
        """
        Call a tool on a specified server.
        """
        session = self.sessions.get(server_name)
        if session is None:
            raise ValueError(f"Server '{server_name}' not connected")
        return await session.call_tool(tool_name, arguments)

    async def process_query(
        self, 
        default_server: str, 
        query: str, 
        ollama_tools: List[Dict],
        merged_tools_list_names: List[str], 
        merged_resources_list_names: List[str],
        messages: Optional[List[Dict]] = None
    ) -> tuple[str, list[dict]]:
        """
        Process a query by sending it to the Ollama API with the merged tool list.
        When a tool call is returned, the function examines the tool name's prefix and
        dispatches the call to the appropriate server.
        """
        if messages is None:
            messages = [{"role": "user", "content": query}]
        else:
            messages.append({"role": "user", "content": query})

        # Prepare the payload for Ollama.
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "3m",
            "options": {'temperature': OLLAMA_TEMPERATURE, 'seed': OLLAMA_SEED},
            "tools": ollama_tools,
        }
        payload = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        # Send the initial query to Ollama.
        response = requests.post(url, headers=headers, data=payload, stream=False)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(response_data)

        tool_results = []
        final_text = []
        assistant_response_content = response_data.get('message', {}).get('content', None)
        if assistant_response_content:
            final_text.append(assistant_response_content)
            logger.info(assistant_response_content)

        tool_calls_content = response_data.get('message', {}).get('tool_calls', None)
        logger.debug(tool_calls_content)

        if tool_calls_content:
            logger.info(f"Tool calls: {len(tool_calls_content)}")
            for call in tool_calls_content:
                inner_dict = call['function']
                full_function_name = inner_dict['name']  # e.g. "pgsql:run_select_query"
                logger.info(f"Call for {full_function_name}")
                arguments = inner_dict['arguments']
                logger.info(f"With Args: {arguments}")

                # Extract the server prefix and tool name.
                if ':' in full_function_name:
                    server_id, tool_name = full_function_name.split(':', 1)
                else:
                    # If no prefix, use the default server.
                    logger.warning("no prefix, use the default server.")
                    server_id, tool_name = default_server, full_function_name

                # Check if the full name exists in the merged tool list.
                if full_function_name in merged_tools_list_names:
                    logger.info(f"Dispatching tool call to server '{server_id}' for tool '{tool_name}'")
                    session = self.sessions.get(server_id)
                    if session is None:
                        logger.error(f"Server '{server_id}' not connected.")
                        answer = {"success": False, "error": f"Server '{server_id}' not connected."}
                    else:
                        # Call the tool using the unprefixed tool name.
                        result = await session.call_tool(tool_name, arguments)
                        answer = json.loads(result.content[-1].text)
                else:
                    logger.error('Function not found in merged tool list')
                    answer = {"success": False, "error": "Invalid function name."}

                tool_results.append({"call": full_function_name, "result": answer})
                final_text.append(f"[Calling tool {full_function_name} with args {arguments}]")

                # Append tool call result to conversation history.
                messages.append({"role": "tool", "content": json.dumps(answer)})
                
                # Get next response from Ollama after tool execution.
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "3m",
                    "options": {'temperature': OLLAMA_TEMPERATURE, 'seed': OLLAMA_SEED},
                }
                response = requests.post(url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                assistant_response_content = response_data.get('message', {}).get('content', None)
                if assistant_response_content:
                    final_text.append(assistant_response_content)
                    messages.append({"role": "assistant", "content": assistant_response_content})
                    print("\n" + YELLOW + assistant_response_content + RESET)

            logger.info('End of Tool Calls loop')
        elif assistant_response_content:
            final_text.append(assistant_response_content)
            messages.append(response_data['message'])

        return "\n".join(final_text), messages

    async def chat_loop(
        self, 
        default_server: str, 
        ollama_tools: List[Dict],
        merged_tools_list_names: List[str], 
        merged_resources_list_names: List[str]
    ):
        """
        Run an interactive chat loop.
        Although a default server is provided, tool calls will be routed based on the namespaced tool name.
        """
        print(f"\nMCP Client for '{default_server}' Started!")
        print("Type your queries or 'quit' to exit.")
        messages = None
        while True:
            query = input("\nQuery => ").strip()
            if query.lower() in ['quit', 'exit']:
                await self.cleanup()
                print('User Exit')
                break
            response, messages = await self.process_query(
                default_server, query, ollama_tools,
                merged_tools_list_names, merged_resources_list_names, messages
            )
            print("\n" + BLUE + messages[-1]['content'] + RESET)

    async def cleanup(self):
        """
        Clean up all connections.
        """
        await self.exit_stack.aclose()


# Global URL and model settings for the Ollama API.
url = f"{OLLAMA_BASE_URL}/api/chat"
system_prompt = None
model = "llama3-groq-tool-use:8b-q8_0"

async def main():
    # Create an instance of the multi-server client.
    client = MultiServerMCPClient()
    
    # Connect to the two MCP servers.
    await client.connect_to_server("domoticz", "mcp_server_domoticz.py")
    await client.connect_to_server("pgsql", "mcp_server_pgsql.py")
    
    # Convert each server's tools to the Ollama format with namespacing.
    ollama_tools_domoticz = client.convert_mcp_tools_to_ollama_tools(client.tools["domoticz"], "domoticz")
    ollama_tools_pgsql = client.convert_mcp_tools_to_ollama_tools(client.tools["pgsql"], "pgsql")
    
    # Merge the two tool lists.
    merged_ollama_tools = ollama_tools_domoticz + ollama_tools_pgsql

    # Merge tool names (with prefixes) for dispatch.
    tools_list_names_domoticz = [f"domoticz:{tool.name}" for tool in client.tools["domoticz"]]
    tools_list_names_pgsql = [f"pgsql:{tool.name}" for tool in client.tools["pgsql"]]
    merged_tools_list_names = tools_list_names_domoticz + tools_list_names_pgsql
    
    # Merge resources lists (if needed).
    resources_list_names_domoticz = [res['name'] for res in client.resources["domoticz"]]
    resources_list_names_pgsql = [res['name'] for res in client.resources["pgsql"]]
    merged_resources_list_names = resources_list_names_domoticz + resources_list_names_pgsql
    
    # Now start the chat loop.
    # Even though we pass "domoticz" as the default server,
    # tool calls are routed based on the tool's namespaced name.
    print("Starting unified chat loop with merged tools:")
    await client.chat_loop("domoticz", merged_ollama_tools, merged_tools_list_names, merged_resources_list_names)
    
    await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        print("Cancelled")
        sys.exit(0)