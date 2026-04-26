"""
MCP Server for Selenium browser control.

Exposes tools: browser_navigate, browser_execute_script, browser_get_html, browser_close.
Run as: python -m web_agent_framework.mcp_servers.mcp_server_selenium
"""
import asyncio
import json
import sys
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

# Selenium driver state (module-level for tool handlers)
_driver = None


def _ensure_driver_sync():
    """Start Selenium driver if not already running (sync, for use in async wrapper)."""
    global _driver  # noqa: PLW0603
    if _driver is not None:
        return
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        _driver = webdriver.Chrome(options=opts)
    except Exception as e:
        raise RuntimeError(f"Failed to start Selenium: {e}")


async def _ensure_driver():
    """Start Selenium driver (run in executor to avoid blocking)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_driver_sync)


def main():
    app = Server("mcp-browser-selenium")

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
        global _driver
        args = arguments or {}

        try:
            if name == "browser_navigate":
                await _ensure_driver()
                url = args.get("url", "")
                if not url:
                    raise ValueError("Missing url")
                _driver.get(url)
                return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps({"ok": True}))])

            elif name == "browser_execute_script":
                await _ensure_driver()
                script = args.get("script", "")
                result = _driver.execute_script(script)
                out = json.dumps(result) if result is not None else "null"
                return types.CallToolResult(content=[types.TextContent(type="text", text=out)])

            elif name == "browser_get_html":
                await _ensure_driver()
                html = _driver.page_source
                return types.CallToolResult(content=[types.TextContent(type="text", text=str(html))])

            elif name == "browser_close":
                if _driver:
                    _driver.quit()
                    _driver = None
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
