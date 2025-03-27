import asyncio
import requests
import json
import urllib.parse
import os
import colorama
import logging
from typing import List, Optional, Dict, Any, Callable, Generator, Sequence, Mapping, Union
from contextlib import AsyncExitStack 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client 
from mcp.types import AnyUrl, Resource, Tool

logging.basicConfig(
    format = '%(asctime)s - %(name)-20s - %(levelname)-10s - %(message)-40s \t (%(filename)s:%(lineno)d)',
    stream = sys.stdout,
    level  = level
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

B_RED        = colorama.Back.RED
RED          = colorama.Fore.RED
BLUE         = colorama.Fore.BLUE
CYAN         = colorama.Fore.CYAN
GREEN        = colorama.Fore.GREEN
YELLOW       = colorama.Fore.YELLOW
MAGENTA      = colorama.Fore.MAGENTA
YELLOW_LIGHT = colorama.Fore.LIGHTYELLOW_EX
RESET        = colorama.Style.RESET_ALL

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_SEED: int = 1234567890

logger = init_logger()

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

class MyMCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama_api_url = OLLAMA_BASE_URL
        logger.info(model)

    def convert_mcp_tools_to_ollama_tools(self, tools) -> List[Dict]:
        logger.info('Convert MCP Tools => Ollama Tools')
        result = []
        for tool in tools:
            function_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.strip(),
                    "parameters": tool.inputSchema
                }
            }
            result.append(function_dict)
        return result

    def convert_mcp_resources_to_ollama_tools(self, resources) -> List[Dict]:
        logger.info('Convert MCP Resources => Ollama Tools')
        result = []
        for item in resources:
            function_dict = {
                "type": "function",
                "function": {
                    "name": item['name'],
                    "description": item.description.strip(),
                    "parameters": {}
                }
            }
            result.append(function_dict)
        return result

    async def connect_to_server(self, server_script_path: str) -> List[Dict]:
        """Connect to an MCP server Args:
            server_script_path: Path to the server script (.py or .js)
        """
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
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        
        # List available resources
        resources_raw = await self.session.list_resources()

        # List available tools
        tools_raw = await self.session.list_tools()
        tools = tools_raw.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        return tools_raw, resources_raw

    async def process_query(self, query: str, ollama_tools: List[Dict], tools_list_names, resources_list_names, messages:list) -> str:
        """Process a query using Ollama and available tools"""

        if messages is None:
            if system_prompt is None:
                messages = [
                    {"role": "user", "content": query}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]

        else:
            messages.append({"role": "user", "content": query})
        # Initial Ollama API call
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

        response = requests.post(url, headers=headers, data=payload, stream=False)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(response_data)

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        assistant_response_content = response_data.get('message', {}).get('content', None)
        if assistant_response_content is not None and assistant_response_content != '':
            final_text.append(assistant_response_content)
            logger.info(assistant_response_content)

        tool_calls_content = response_data.get('message', {}).get('tool_calls', None)
        logger.debug(tool_calls_content)

        if tool_calls_content is not None:
            logger.info(f"Tool calls: {len(tool_calls_content)}")
            for call in tool_calls_content:
                inner_dict = call['function']
                function_name = inner_dict['name']
                logger.info(f"Call for {function_name}")
                arguments = inner_dict['arguments']
                logger.info(f"With Args: {arguments}")

                # Check if the variable is in the list of functions
                is_in_tools_list = any(func == function_name for func in tools_list_names)
                logger.debug(f"Is in Tool List ? {is_in_tools_list}")
                is_in_resources_list = any(item == function_name for item in resources_list_names)
                logger.debug(f"Is in Resource List ? {is_in_resources_list}")

                if is_in_tools_list:
                    logger.info('Function exist : Tool')
                    logger.info('Execute Tool call')
                    result = await self.session.call_tool(function_name, arguments)
                    answer = json.loads(result.content[-1].text)

                elif is_in_resources_list:
                    logger.info('Function exist : Resource')
                    logger.info('Execute Resource call')
                    content, result = await self.session.read_resource(any(item['uri'] == function_name for item in resources_list_names))
                    logger.info(urllib.parse.unquote(result[1][-1].text))
                    answer = json.loads({"success": True, "message": urllib.parse.unquote(result[1][-1].text)})

                else:
                    logger.error('Function IS NOT IN BOTH LIST')
                    answer = {"success": False, "error" : "This function doesn't exist. Please try again with a valid function name."}

                # Execute tool call
                tool_results.append({"call": function_name, "result": answer})
                final_text.append(f"[Calling tool {function_name} with args {arguments}]")
            
                # Continue conversation with tool results
                if answer.get('success'):
                    logger.info('Tool call success')
                    messages.append({
                    "role": "tool",
                    "content": json.dumps(answer)
                    })
                else:
                    logger.warning('Tool call failed')
                    messages.append({    ### if error ?
                        "role": "tool",
                        "content": json.dumps(answer)
                    })

                # Get next response from Ollama
                logger.info('Next response')
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    #"format": "json",
                    "keep_alive": "3m",
                    "options": {'temperature': OLLAMA_TEMPERATURE, 'seed': OLLAMA_SEED},
                }

                response = requests.post(url, json=payload)
                response.raise_for_status()
                response_data = response.json()

                assistant_response_content = response_data.get('message', {}).get('content', None)
                if assistant_response_content is not None:
                    final_text.append(assistant_response_content)
                    messages.append({"role": "assistant", "content": assistant_response_content})     ## this add Assistant result to chat messages
                    print("\n" + YELLOW + assistant_response_content + RESET)

            logger.info('End of Tool Calls For Loop')

        elif assistant_response_content is not None:
            logger.info('assistant response not None')
            if final_text != []:
                logger.info('final text not empty')
                if final_text[-1] != assistant_response_content:
                    final_text.append(assistant_response_content)
                    print("\n" + MAGENTA + assistant_response_content + RESET)

            logger.info('Append Tool Output to Chat History')
            messages.append(response_data['message']) ## this add tool result to chat messages

        return "\n".join(final_text), messages

    async def chat_loop(self, ollama_tools, tools_list_names, resources_list_names):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        messages = None
        while True:
            query = input("\nQuery => ")
            query = query.strip()
            if query.lower() == 'quit' or query.lower() == 'exit':
                await self.cleanup()
                print('\n')
                if messages is not None:
                    for msg in messages:
                        print(msg)
                        print('\n')
                print('User Exit')
                break
            if messages is not None:
                logger.info(f"Messages : {len(messages)}")
            response, messages = await self.process_query(query, ollama_tools, tools_list_names, resources_list_names, messages)
            #print("\n" + response)                               ## include intermediate steps
            print("\n" + BLUE + messages[-1]['content'] + RESET)  ## last message

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    global system_prompt
    if len(sys.argv) < 2:
        print("Alternative usage: python client.py <path_to_server_script>")
        server_file = 'ollama_mcp_server.py'

    else:
        server_file = sys.argv[1]

    client = MyMCPClient()

    try:
        mcp_tools_list, mcp_resources_list = await client.connect_to_server(server_file)
        mcp_resources_list = build_resource_list(mcp_resources_list)

        ollama_tools = client.convert_mcp_tools_to_ollama_tools(mcp_tools_list.tools)

        tools_list_names = [tool.name for tool in mcp_tools_list.tools]
        logger.info(tools_list_names)

        resources_list_names = [resource['name'] for resource in mcp_resources_list]
        logger.info(resources_list_names)
        await client.chat_loop(ollama_tools, tools_list_names, resources_list_names)

    except KeyboardInterrupt:
        logger.info('User Exit')

    finally:
        await client.cleanup()

url = f"{OLLAMA_BASE_URL}/api/chat"

system_prompt = None

#model = "qwen2.5:14b-instruct"
#model = "qwen2.5:7b-instruct-q8_0"
#model = "aya-expanse:8b-q8_0"
#model = "huihui_ai/dolphin3-abliterated:8b-llama3.1-q8_0"  # ok
model = "llama3-groq-tool-use:8b-q8_0"

if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except asyncio.CancelledError: # instead of KeyboardInterrupt
        print("Cancelled")
        exit(0)