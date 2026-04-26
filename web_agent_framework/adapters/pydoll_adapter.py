import asyncio
from typing import Any, Optional
from pydoll.browser import Chrome
from .browser import BrowserAdapter, PageAdapter

class PydollPageAdapter:
    def __init__(self, tab):
        self.tab = tab

    async def navigate(self, url: str) -> None:
        await self.tab.go_to(url)

    async def execute_script(self, script: str, *args) -> Any:
        # Pydoll's execute_script might need adjustment if it doesn't support *args directly
        return await self.tab.execute_script(script)

    async def get_html(self) -> str:
        html = await self.tab.execute_script("return document.documentElement.outerHTML;")
        return str(html)

    async def screenshot(self, path: str) -> None:
        # Check if pydoll tab has a screenshot method, otherwise implement via CDP
        if hasattr(self.tab, 'screenshot'):
            await self.tab.screenshot(path)
        else:
            # Fallback or raise not implemented
            print("[PydollAdapter] Screenshot not directly implemented in tab")

    async def click(self, selector: str) -> None:
        # Implement click using JS if not native
        script = f"document.querySelector('{selector}').click();"
        await self.tab.execute_script(script)

    async def fill(self, selector: str, value: str) -> None:
        # Implement fill using JS if not native
        script = f"""
        (function() {{
            const el = document.querySelector('{selector}');
            if (el) {{
                el.value = '{value}';
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}
        }})();
        """
        await self.tab.execute_script(script)

class PydollBrowserAdapter:
    def __init__(self):
        self.browser = None

    async def start(self, **kwargs) -> Any:
        self.browser = Chrome()
        # Chrome() in pydoll might be a context manager, but here we use it as an object
        # Based on bootstrap.py: async with Chrome() as browser: tab = await browser.start()
        # To persist the browser, we might need a different approach if Chrome() only works as CM.
        # For now, let's assume we can start it.
        self.tab = await self.browser.start()
        return PydollPageAdapter(self.tab)

    async def close(self) -> None:
        if self.browser:
            await self.browser.close()

    async def get_page(self, url: str) -> Any:
        if not self.tab:
            await self.start()
        await self.tab.go_to(url)
        return PydollPageAdapter(self.tab)
