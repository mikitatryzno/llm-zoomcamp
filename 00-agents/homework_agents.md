**Setup Environment**

```python
import random
known_weather_data = {
    'berlin': 20.0
}
def get_weather(city: str) -> float:
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]
    return round(random.uniform(-5, 35), 1)
```


**Q1. Define function description**

We need to create a proper function description for the get_weather function to use it as a tool for our agent:

```python
get_weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current temperature for a specified city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city for which to retrieve the temperature"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }
}
```
The answer for TODO3 is `city`.

**Q2. Adding another tool**

Let's create a description for the set_weather function:

```python
set_weather_tool = {
    "type": "function",
    "name": "set_weather",
    "description": "Set the temperature for a specified city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city for which to set the temperature"
            },
            "temp": {
                "type": "number",
                "description": "The temperature value to set for the city"
            }
        },
        "required": ["city", "temp"],
        "additionalProperties": False
    }
}
```

The description can be `The temperature value to set for the city`

***Q3. Install FastMCP**

To install FastMCP, run:

```bash
pip install fastmcp
```

To check the version of FastMCP installed:

```python
import fastmcp
print(fastmcp.__version__)
```

The version is `2.10.4`

**Q4. Simple MCP Server**
Let's create a weather server using FastMCP:

```python
# weather_server.py
import random
from fastmcp import FastMCP
known_weather_data = {
    'berlin': 20.0
}
mcp = FastMCP("Demo 🚀")
@mcp.tool
def get_weather(city: str) -> float:
    """
    Retrieves the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        float: The temperature associated with the city.
    """
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    return round(random.uniform(-5, 35), 1)


@mcp.tool
def set_weather(city: str, temp: float) -> str:
    """
    Sets the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to set the weather data.
        temp (float): The temperature to associate with the city.

    Returns:
        str: A confirmation string 'OK' indicating successful update.
    """
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'

if __name__ == "__main__":
    mcp.run()
```

When you run this script, you'll see output like:
```
Starting MCP server 'Demo 🚀' with transport 'stdio'
```
The answer for the TODO is `stdio`.

**Q5. Protocol**

To interact with the MCP server, we follow the JSON-RPC protocol:
Send initialization request:

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": true}, "sampling": {}}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}
```

Get acknowledgment from server.
Confirm initialization:
```json
{"jsonrpc": "2.0", "method": "notifications/initialized"}
```

Request available methods:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
```
Ask for temperature in Berlin:
```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_weather", "arguments": {"city": "Berlin"}}}
```
The response would be something like:
```json
{"jsonrpc":"2.0","id":3,"result":20.0}
```

***Q6. Client**

Using the provided mcp_client.py, let's implement an MCP client to interact with our server:

```python
# client_script.py
import asyncio
from fastmcp import Client
import weather_server  # Import our weather server module
# For Jupyter notebook
async def main():
    async with Client(weather_server.mcp) as mcp_client:
        tools = await mcp_client.tools.list()
        print(tools)
# For running as a script
if __name__ == "__main__":
    asyncio.run(main())
```    

The output will look something like:

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Retrieves the temperature for a specified city.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city for which to retrieve weather data."
          }
        },
        "required": ["city"]
      },
      "outputSchema": {
        "type": "number",
        "description": "The temperature associated with the city."
      }
    },
    {
      "name": "set_weather",
      "description": "Sets the temperature for a specified city.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city for which to set the weather data."
          },
          "temp": {
            "type": "number",
            "description": "The temperature to associate with the city."
          }
        },
        "required": ["city", "temp"]
      },
      "outputSchema": {
        "type": "string",
        "description": "A confirmation string 'OK' indicating successful update."
      }
    }
  ]
}
```

Summary of Answers
Q1: The value for TODO3 is "city".
Q2: A complete function description for set_weather was provided above.
Q3: The version of FastMCP will depend on when you install it (e.g., 1.9.4).
Q4: The transport used in the MCP server output is stdio.
Q5: The response for the Berlin temperature query would be {"jsonrpc":"2.0","id":3,"result":20.0}.
Q6: The output shows the list of available tools with their descriptions, input schemas, and output schemas as shown above.


**Optional: Using MCP with chat_assistant.py**
If you want to integrate the MCP server with the chat assistant:

Download the chat_assistant.py file:
```bash
wget https://raw.githubusercontent.com/alexeygrigorev/rag-agents-workshop/refs/heads/main/chat_assistant.py
```

Use the provided MCPTools class from mcp_client.py to integrate with the chat assistant:
```python
import mcp_client
import chat_assistant
from openai import OpenAI
# Initialize OpenAI client
client = OpenAI()
# Initialize the MCP client
our_mcp_client = mcp_client.MCPClient(["python", "weather_server.py"])
our_mcp_client.start_server()
our_mcp_client.initialize()
our_mcp_client.initialized()
# Create MCP tools wrapper
mcp_tools = mcp_client.MCPTools(mcp_client=our_mcp_client)
# Set up the chat assistant
developer_prompt = """
You help users find out the weather in their cities. 
If they didn't specify a city, ask them. Make sure we always use a city.
""".strip()
chat_interface = chat_assistant.ChatInterface()
chat = chat_assistant.ChatAssistant(
    tools=mcp_tools,
    developer_prompt=developer_prompt,
    chat_interface=chat_interface,
    client=client
)
# Run the chat assistant
chat.run()
```