# Ollama MCP

## Description
MCP Implementation for Ollama. Single and Multi Server.<br>
Here, 'MCP Server' is a python script, which is started by the MCP Client.

## Usage
[Ollama](https://ollama.com/) need to be installed and running somewhere.

Start Client with default MCP Server hardcoded in the script :
```shell
python ollama_mcp_mono.py
```

Start Client with specified MCP Server as an Arg :
```shell
python ollama_mcp_mono.py my_mcp_server.py
```

Start Client with multi MCP Server hardcoded in the script :
```shell
python ollama_mcp_multi.py
```

## Authors and acknowledgment
Special thanks to Eric Hartford for his project [Dolphin-MCP](https://github.com/cognitivecomputations/dolphin-mcp).

## License
Do what you want.

## Project status
This is mainly a POC and it should not be updated.
