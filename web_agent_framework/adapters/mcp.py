"""
MCP adapters for browser control via Model Context Protocol.

- Default: Our own MCP server runs (Pydoll or Selenium). User selects via MCP_BROWSER (default: pydoll).
- MCP_SERVER_URL: Optional override to connect to an external MCP server instead.
"""

import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Protocol, Any, List, Dict, Optional


class MCPAdapter(Protocol):
    """
    Protocol for interacting with browser instances via Multi-Client Protocol (MCP).
    This allows the agent to drive browsers hosted in separate processes or servers.
    """
    async def connect(self, server_url: str) -> None:
        ...

    async def disconnect(self) -> None:
        ...

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        ...


class MCPPageAdapter:
    """Page adapter that delegates to MCP tool calls."""

    def __init__(self, mcp_adapter: "MCPBrowserAdapter", url: str):
        self.mcp = mcp_adapter
        self.url = url

    async def navigate(self, url: str) -> None:
        result = await self.mcp.execute_tool("browser_navigate", {"url": url})
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error", "Navigate failed"))
        self.url = url

    async def execute_script(self, script: str, *args) -> Any:
        result = await self.mcp.execute_tool(
            "browser_execute_script",
            {"script": script, "args": list(args)},
        )
        if isinstance(result, dict):
            return result.get("result", result.get("content", result))
        return result

    async def get_html(self) -> str:
        result = await self.mcp.execute_tool("browser_get_html", {})
        if isinstance(result, dict):
            return result.get("html", "") or result.get("content", str(result))
        return str(result) if result is not None else ""


def _get_internal_mcp_server_command() -> tuple[str, list[str], str]:
    """Return (command, args, cwd) to spawn our Pydoll or Selenium MCP server.
    Uses -u (unbuffered) and -m (module) for reliable stdio and imports.
    """
    from ..config import MCP_BROWSER, WORKSPACE_ROOT
    module = f"web_agent_framework.mcp_servers.mcp_server_{MCP_BROWSER}"
    src_dir = WORKSPACE_ROOT / "src"
    if not (src_dir / "web_agent_framework").exists():
        src_dir = WORKSPACE_ROOT  # fallback if structure differs
    return sys.executable, ["-u", "-m", module], str(src_dir)


class MCPBrowserAdapter:
    """
    Browser adapter that uses MCP server for browser control.
    - If MCP_SERVER_URL is set: connect to that external server.
    - Else: spawn our own MCP server (Pydoll or Selenium per MCP_BROWSER).
    """

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = (server_url or os.getenv("MCP_SERVER_URL", "")).strip()
        self._session = None
        self._page = None
        self._exit_stack: Optional[AsyncExitStack] = None

    async def _ensure_session(self) -> bool:
        """Connect to MCP server. Returns True if connected."""
        if self._session:
            return True
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            if self.server_url:
                # External MCP server (e.g. MCP_SERVER_URL="stdio:npx -y @modelcontextprotocol/server-browser")
                url = self.server_url
                if url.startswith("stdio:"):
                    rest = url[6:].strip().split()
                    cmd = rest[0] if rest else "npx"
                    args = rest[1:] if len(rest) > 1 else (["-y", "@modelcontextprotocol/server-browser"] if cmd == "npx" else [])
                else:
                    cmd, args = "npx", ["-y", "@modelcontextprotocol/server-browser"]
                params = StdioServerParameters(command=cmd, args=args)
            else:
                # Our own MCP server (Pydoll or Selenium)
                cmd, args, cwd = _get_internal_mcp_server_command()
                params = StdioServerParameters(command=cmd, args=args, cwd=cwd)

            # stdio_client and ClientSession are async context managers; use AsyncExitStack
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = stdio_transport
            self._session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self._session.initialize()
            return True
        except ImportError:
            print("[MCP] mcp package not installed. Run: pip install mcp")
            return False
        except Exception as e:
            print(f"[MCP] Connect failed: {e}")
            return False

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an MCP tool and return the result."""
        if not await self._ensure_session():
            return {"ok": False, "error": "MCP not available"}
        try:
            result = await self._session.call_tool(tool_name, arguments)
            if hasattr(result, "content") and result.content:
                for c in result.content:
                    if hasattr(c, "text") and c.text:
                        try:
                            return json.loads(c.text)
                        except json.JSONDecodeError:
                            return {"ok": True, "content": c.text}
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def start(self, **kwargs) -> Any:
        """Start browser session. Returns page adapter."""
        if await self._ensure_session():
            self._page = MCPPageAdapter(self, "")
            return self._page
        raise RuntimeError("MCP not available. Use PydollBrowserAdapter.")

    async def close(self) -> None:
        if self._session:
            try:
                await self.execute_tool("browser_close", {})
            except Exception:
                pass
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                pass
            self._exit_stack = None
        self._session = None


class PydollMCPAdapter:
    """Legacy name: MCP adapter that delegates to Pydoll when MCP unavailable."""

    def __init__(self):
        self.server_url = None
        self.client = None

    async def connect(self, server_url: str) -> None:
        self.server_url = server_url
        print(f"[PydollMCP] Connecting to {server_url}...")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        print(f"[PydollMCP] Executing tool: {tool_name} with {arguments}")
        return {"ok": True, "result": "mocked result"}


class SeleniumMCPAdapter:
    """Legacy MCP adapter for Selenium."""

    def __init__(self):
        self.server_url = None

    async def connect(self, server_url: str) -> None:
        self.server_url = server_url
        print(f"[SeleniumMCP] Connecting to {server_url}...")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        print(f"[SeleniumMCP] Executing tool: {tool_name}")
        return {"ok": True, "result": "mocked result"}


def get_browser_adapter():
    """
    Factory: returns the browser adapter based on config.

    - MCP_SERVER_URL set: MCPBrowserAdapter (connects to external MCP server)
    - Else: MCPBrowserAdapter (spawns our own MCP server — Pydoll or Selenium per MCP_BROWSER, default pydoll)

    Chroma DB (DB_DIR) is used by RAG for Pydoll/Selenium context — see config.py.
    """
    return MCPBrowserAdapter()
