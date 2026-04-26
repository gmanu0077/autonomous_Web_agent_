from typing import Protocol, Any, Optional

class BrowserAdapter(Protocol):
    async def start(self, **kwargs) -> Any:
        ...
    
    async def close(self) -> None:
        ...

    async def get_page(self, url: str) -> Any:
        ...

class PageAdapter(Protocol):
    async def navigate(self, url: str) -> None:
        ...

    async def execute_script(self, script: str, *args) -> Any:
        ...

    async def get_html(self) -> str:
        ...

    async def screenshot(self, path: str) -> None:
        ...

    async def click(self, selector: str) -> None:
        ...

    async def fill(self, selector: str, value: str) -> None:
        ...
