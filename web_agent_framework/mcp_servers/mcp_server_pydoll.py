"""
MCP Server for Pydoll browser control.

Exposes tools: browser_navigate, browser_execute_script, browser_get_html, browser_close.
Run as: python -m web_agent_framework.mcp_servers.mcp_server_pydoll
"""
import json
import sys


def _extract_cdp_value(raw):
    """Extract actual value from CDP response. Pydoll may return {result: {result: {type, value}}} for long strings."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return raw
    if "result" in raw:
        raw = raw["result"]
    if isinstance(raw, dict) and "result" in raw:
        raw = raw["result"]
    if isinstance(raw, dict) and (raw.get("objectId") or raw.get("type") == "function"):
        return {"_error": "Script returned non-serializable value (function/object). Use IIFE + JSON.stringify: return JSON.stringify((function(){ ... return result; })());"}
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _wrap_script_for_serialization(script: str) -> str:
    """
    Wrap user script so we always get a JSON string back from CDP.
    Fixes 'non-serializable' errors when Pydoll/CDP returns objectId instead of value.
    User script has 'return X' - we run it as (function(){ return X })() and stringify if needed.
    """
    script_escaped = json.dumps("(function(){" + script + "})()")
    return (
        "return (function(){"
        "try{"
        "var __r=eval(" + script_escaped + ");"
        "return (typeof __r==='string')?__r:JSON.stringify(__r);"
        "}catch(e){"
        "return JSON.stringify({'_error':'Script error: '+e.toString()});"
        "}"
        "})();"
    )
from pathlib import Path

# Add project root to path
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Pydoll browser state (module-level for tool handlers)
_browser = None
_tab = None


async def _ensure_browser():
    """Start Pydoll browser if not already running."""
    global _browser, _tab
    if _tab is not None:
        return
    try:
        from pydoll.browser import Chrome
        _browser = Chrome()
        _tab = await _browser.start()
    except Exception as e:
        raise RuntimeError(f"Failed to start Pydoll: {e}")


def main():
    app = Server("mcp-browser-pydoll")

    @app.list_tools()
    async def handle_list_tools() -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="browser_navigate",
                    title="Navigate to URL",
                    description="Navigate the browser to a URL",
                    inputSchema={
                        "type": "object",
                        "required": ["url"],
                        "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                    },
                ),
                types.Tool(
                    name="browser_execute_script",
                    title="Execute JavaScript",
                    description="Execute JavaScript in the page and return the result",
                    inputSchema={
                        "type": "object",
                        "required": ["script"],
                        "properties": {
                            "script": {"type": "string", "description": "JavaScript to execute"},
                            "args": {"type": "array", "description": "Optional arguments"},
                        },
                    },
                ),
                types.Tool(
                    name="browser_get_html",
                    title="Get Page HTML",
                    description="Return the full HTML of the current page",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="browser_close",
                    title="Close Browser",
                    description="Close the browser session",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        )

    @app.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        global _browser, _tab
        args = arguments or {}

        try:
            if name == "browser_navigate":
                await _ensure_browser()
                url = args.get("url", "")
                if not url:
                    raise ValueError("Missing url")
                await _tab.go_to(url)
                return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps({"ok": True}))])

            elif name == "browser_execute_script":
                await _ensure_browser()
                script = args.get("script", "")
                # Wrap script to force JSON string return — avoids CDP objectId serialization failures
                wrapped = _wrap_script_for_serialization(script)
                result = await _tab.execute_script(wrapped)
                result = _extract_cdp_value(result) if isinstance(result, dict) else result
                out = result if isinstance(result, str) else json.dumps(result)
                return types.CallToolResult(content=[types.TextContent(type="text", text=out)])

            elif name == "browser_get_html":
                await _ensure_browser()
                html = await _tab.execute_script("return document.documentElement.outerHTML;")
                html = _extract_cdp_value(html) if isinstance(html, dict) else html
                return types.CallToolResult(content=[types.TextContent(type="text", text=str(html or ""))])

            elif name == "browser_close":
                if _browser:
                    await _browser.close()
                    _browser, _tab = None, None
                return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps({"ok": True}))])

            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps({"ok": False, "error": str(e)}))],
                isError=True,
            )

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
