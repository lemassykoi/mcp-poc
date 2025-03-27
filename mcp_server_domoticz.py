import os
import json
import aiohttp
from contextlib import asynccontextmanager
from typing import AsyncIterator
from mcp.server.fastmcp import FastMCP, Context
from mcp.server import Server

domoticz_user      = 'mcp_client'
domoticz_pass      = 'password'
domoticz_ip        = '10.0.0.1'
domoticz_port      = '80'
domoticz_base_url  = f'http://{domoticz_user}:{domoticz_pass}@{domoticz_ip}:{domoticz_port}'

# -------------------------------------------------------------------
# Domoticz Configuration (use environment variables or defaults)
# -------------------------------------------------------------------
DOMOTICZ_HOST = os.environ.get("DOMOTICZ_HOST", domoticz_base_url)
DOMOTICZ_API_PATH  = "/json.htm"
DOMOTICZ_PLAN_NAME = "LLM"
## For large devices number setup, create a Plan in Domoticz, and assign the
## devices you want to be able to read with MCP. This will filter out
## the get_devices result which is about 500k chars for me.

# -------------------------------------------------------------------
# Lifespan: Create an aiohttp session to be shared by the tools.
# -------------------------------------------------------------------
@asynccontextmanager
async def app_lifespan(server: Server) -> AsyncIterator[dict]:
    session = aiohttp.ClientSession()
    try:
        # The lifespan context contains our aiohttp session.
        yield {"session": session}
    finally:
        await session.close()


# Create the FastMCP instance with lifespan support.
mcp = FastMCP(
    "Domoticz MCP Server",
    instructions = "You are a Domoticz senior expert. You are in charge of a Home Automation System instance, running Domoticz.",
    lifespan     = app_lifespan
)

# -------------------------------------------------------------------
# Utility: Asynchronously query the Domoticz JSON API.
# -------------------------------------------------------------------
async def domoticz_query(session: aiohttp.ClientSession, params: dict) -> dict:
    url = f"{DOMOTICZ_HOST}{DOMOTICZ_API_PATH}"
    try:
        async with session.get(url, params=params) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# Tool: List Devices
# -------------------------------------------------------------------
@mcp.tool()
async def list_devices(ctx: Context) -> dict:
    """
    Retrieves devices from the Domoticz instance that belong to the plan named "LLM".
    Devices are filtered by checking if the plan idx (from getplans) is included in their PlanIDs.
    Only key data fields are returned.
    """
    session = ctx.request_context.lifespan_context["session"]

    # 1. Retrieve plans to get the idx for the "LLM" plan.
    plan_params = {
         "type": "command",
         "param": "getplans",
         "order": "name",
         "used": "true"
    }
    response_plans = await domoticz_query(session, plan_params)
    json_plans = response_plans.get("result", [])
    # Map plan names to idx values.
    plan_mapping = {plan["Name"]: int(plan["idx"]) for plan in json_plans if "Name" in plan and "idx" in plan}
    if DOMOTICZ_PLAN_NAME not in plan_mapping:
         return json.dumps({"error": f"Plan '{DOMOTICZ_PLAN_NAME}' not found."})
    plan_idx = plan_mapping[DOMOTICZ_PLAN_NAME]

    # 2. Retrieve all devices.
    device_params = {
         "type": "devices",
         "filter": "all",
         "used": "true",
         "order": "Name"
    }
    response_devices = await domoticz_query(session, device_params)
    json_devices = response_devices.get("result", [])

    # 3. Filter functions.
    def filter_device_from_plan(device: dict, plan_idx: int):
         # Ensure device has a PlanIDs field (expected to be a list).
         if "PlanIDs" in device and isinstance(device["PlanIDs"], list):
              if plan_idx in device["PlanIDs"]:
                   return device
         return None

    def filter_device_data(device: dict):
         return {
              "Name": device.get("Name"),
              "idx": device.get("idx"),
              "Data": device.get("Data"),
              "SwitchType": device.get("SwitchType")
         }

    # 4. Filter devices that belong to the specified plan.
    filtered_devices_from_plan = [
         filter_device_from_plan(device, plan_idx) for device in json_devices
         if filter_device_from_plan(device, plan_idx) is not None
    ]
    filtered_devices = [
         filter_device_data(device) for device in filtered_devices_from_plan
         if filter_device_data(device) is not None
    ]

    return json.dumps({"success": True, "message": filtered_devices})


# -------------------------------------------------------------------
# Tool: Get Device Details
# -------------------------------------------------------------------
@mcp.tool()
async def get_device(ctx: Context, idx: int) -> dict:
    """
    Gets details for a specific device from Domoticz using its idx.
    """
    session = ctx.request_context.lifespan_context["session"]
    params = {
        "type": "devices",
        "rid": idx
    }
    result = await domoticz_query(session, params)
    return json.dumps(result)

# -------------------------------------------------------------------
# Tool: Switch a Device
# -------------------------------------------------------------------
@mcp.tool()
async def switch_device(ctx: Context, idx: int, action: str) -> dict:
    """
    Switches a device's state. Valid actions are "on", "off", or "toggle".
    """
    if action.lower() not in ["on", "off", "toggle"]:
        return json.dumps({"error": "Invalid action. Use 'on', 'off', or 'toggle'."})
    session = ctx.request_context.lifespan_context["session"]
    switchcmd = action.capitalize()  # Domoticz expects capitalized commands.
    params = {
        "type": "command",
        "param": "switchlight",
        "idx": idx,
        "switchcmd": switchcmd
    }
    result = await domoticz_query(session, params)
    if result['status'] == 'OK':
        return json.dumps({"success": True, "message": result})
    else:
        return json.dumps({"success": False, "message": result})

# -------------------------------------------------------------------
# Prompts: Provide example prompts for LLM integration
# -------------------------------------------------------------------
@mcp.prompt()
def list_devices_prompt() -> str:
    return "List all devices available from the Domoticz server."

@mcp.prompt()
def get_device_prompt(idx: int) -> str:
    return f"Retrieve the details for the device with index {idx}."

@mcp.prompt()
def switch_device_prompt(idx: int, action: str) -> str:
    return f"Switch the device with index {idx} to {action} using the Domoticz API."

# -------------------------------------------------------------------
# Run the MCP server.
# -------------------------------------------------------------------
if __name__ == '__main__':
    mcp.run()