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