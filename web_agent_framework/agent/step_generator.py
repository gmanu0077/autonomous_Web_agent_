import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from ..config import CHAT_MODEL_NAME, TOP_K
from ..core.utils import write_text
from ..adapters.llm import LLMAdapter

# Max chars for RAG (Pydoll code) context in step prompts
RAG_CONTEXT_MAX_CHARS = 4000


class StepGenerator:
    def __init__(self, llm_adapter: LLMAdapter, rag_adapter: Optional[Any] = None):
        self.llm = llm_adapter
        self.rag = rag_adapter
        self.system_prompt = self._load_step_system_prompt()
        self.forbidden_tokens = [
            "pydoll.commands", "from pydoll.commands", "import pydoll.commands",
            "query_selector", "query_selector_all", ".get_attribute(",
            ".get_text(", ".inner_text(", ".text_content(", "page.goto(",
            "tab.go_to(", "playwright", "selenium",
        ]

    def _load_step_system_prompt(self) -> str:
        prompt_path = Path(__file__).parent / "step_system_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "You are an AI assistant that generates Python code for web automation steps using a Pydoll 'tab' object. Generate an 'async def run_step(tab, shared):' function."

    def _validate_generated_step_code(self, step_id: str, code: str) -> Tuple[bool, str]:
        if not code or not code.strip():
            return False, "Empty code."
        if "async def run_step" not in code:
            return False, "Missing required 'async def run_step(tab, shared):' function."
        
        lowered = code.lower()
        for tok in self.forbidden_tokens:
            if tok.lower() in lowered:
                return False, f"Contains forbidden token/API: {tok}"
        return True, "OK"

    def _get_rag_context(self, step_id: str, goal: str, html_snippet: str) -> str:
        """Retrieve Pydoll/Selenium code examples from Chroma via RAG."""
        if not self.rag:
            return "(No RAG context — run build_embeddings to populate Chroma with Pydoll code)"
        query = f"STEP_ID: {step_id}\nGOAL: {goal}\n{html_snippet[:500]}"
        chunks = self.rag.get_top_chunks(query, n_results=TOP_K)
        if not chunks:
            return "(No matching Pydoll code in Chroma — run build_embeddings)"
        return self.rag.build_context_from_chunks(chunks, max_chars=RAG_CONTEXT_MAX_CHARS)

    def _build_dom_chunks_text(self, dom_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build selector/hint text from DOM chunks for step prompts. Includes class-based selectors for e-commerce."""
        if not dom_chunks:
            return ""
        lines = []
        seen = set()
        for c in dom_chunks[:25]:
            sel = c.get("selector")
            hint = c.get("semantic_hint")
            tag = c.get("tag", "?")
            attrs = c.get("attrs") or {}
            aid = attrs.get("id")
            acls = attrs.get("class")
            if isinstance(acls, list):
                acls = " ".join(acls) if acls else ""
            elif not isinstance(acls, str):
                acls = ""
            sel = sel or (f"#{aid}" if aid and isinstance(aid, str) and all(x.isalnum() or x in "-_" for x in aid) else None)
            sel = sel or (tag if tag and tag.startswith("ytd-") else None)
            if not sel and acls:
                first_cls = acls.split()[0] if acls.split() else None
                if first_cls and all(x.isalnum() or x in "-_" for x in first_cls):
                    sel = f"{tag}.{first_cls}"
            if not sel and not hint and not acls:
                continue
            key = (sel or "", hint or "", str(acls)[:50])
            if key in seen:
                continue
            seen.add(key)
            parts = [f"tag={tag}"]
            if sel:
                parts.append(f"selector={sel}")
            if hint:
                parts.append(f"hint={hint}")
            if acls:
                parts.append(f"class={str(acls)[:100]}")
            harvested = c.get("harvested_urls") or []
            if harvested and len(harvested) <= 3:
                parts.append(f"urls_sample={harvested[:3]}")
            lines.append("  - " + ", ".join(parts))
        if not lines:
            return ""
        return "DOM CHUNKS (from page analysis — prefer these selectors in your code):\n" + "\n".join(lines)

    def _build_graph_query_text(self, graph_query_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Build prompt text from graph query result (requirement-driven).
        Use this instead of raw chunks when available — it contains only what the step needs.
        """
        if not graph_query_result:
            return ""
        selectors = graph_query_result.get("selectors") or []
        data_schema = graph_query_result.get("data_schema") or ""
        selectors_for = graph_query_result.get("selectors_for") or ""
        sample_nodes = graph_query_result.get("sample_nodes") or []
        child_hint = graph_query_result.get("child_selector_hint")

        if not selectors and not data_schema:
            return ""

        lines = []
        if selectors:
            lines.append("SELECTORS (from graph query — use these for this step):")
            for s in selectors[:15]:
                lines.append(f"  - {s}")
        if data_schema:
            lines.append(f"\nDATA SCHEMA (expected output shape): {data_schema}")
        if selectors_for:
            lines.append(f"\nQUERY TARGET: {selectors_for}")
        if child_hint:
            lines.append(f"\nCHILD SELECTOR (use inside container): {child_hint}")
        if graph_query_result.get("used_empty_fallback"):
            lines.append("\nNOTE: Graph query had no strong matches; using structural fallbacks. Prefer container + childSelector pattern.")
        if sample_nodes:
            lines.append("\nSAMPLE NODES (structure from page):")
            for n in sample_nodes[:5]:
                parts = [f"tag={n.get('tag')}"]
                if n.get("selector"):
                    parts.append(f"selector={n['selector']}")
                if n.get("hint"):
                    parts.append(f"hint={n['hint']}")
                lines.append("  - " + ", ".join(parts))

        return "\n".join(lines) if lines else ""

    async def generate_step(
        self,
        step_id: str,
        goal: str,
        context: str,
        shared: Dict[str, Any],
        repair_mode: bool = False,
        previous_error: str = "",
        dom_chunks: Optional[List[Dict[str, Any]]] = None,
        user_demand: str = "",
        graph_query_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate step code. Prefers graph_query_result (requirement-driven) over raw dom_chunks.
        Flow: requirements → query graph → code with that data.
        """
        print(f"[StepGenerator] Generating code for step: {step_id} (repair={repair_mode})")

        # Prefer graph query result (requirement-driven) over raw chunks
        dom_section = self._build_graph_query_text(graph_query_result)
        if not dom_section and dom_chunks:
            dom_section = self._build_dom_chunks_text(dom_chunks)
        rag_context = self._get_rag_context(step_id, goal, context)

        prompt = f"""
STEP_ID: {step_id}
STEP_GOAL: {goal}
"""
        if dom_section:
            prompt += f"""
=== GRAPH QUERY RESULT (REQUIREMENT-DRIVEN — USE THESE) ===
The following was queried from the page graph for this step. Use these selectors and data schema — do NOT invent selectors.

{dom_section}

MANDATORY: Use document.querySelector(selector) or querySelectorAll with the selectors above. If a selector does not match, try the next one. Do NOT use generic selectors without combining with the page structure above.

"""
        prompt += f"""
PYDOL CODE PATTERNS (use tab.execute_script, return JSON via IIFE):
{rag_context}

PAGE HTML (supplementary — DOM chunks above take precedence):
{context[:3500]}

CURRENT SHARED STATE (partial):
{json.dumps({k: v for k, v in shared.items() if isinstance(v, (str, int, float, bool, list, dict))}, indent=2)[:1500]}

Generate the Python code for this step. It must contain:
async def run_step(tab, shared):
    ...

CRITICAL: Use JSON.stringify in JS so CDP returns a string (avoids "non-serializable" errors). Example:
  js = r'return JSON.stringify((function(){{ const data = {{}}; /* ... */ return result; }})());'
  raw = await tab.execute_script(js)
  result = json.loads(raw) if isinstance(raw, str) else raw
"""
        if repair_mode:
            prompt += f"\n\nREPAIR MODE: Previous attempt failed with error:\n{previous_error}\nPlease fix the code."

        # In a real implementation, we'd use LLM to generate.
        # For BUILTIN steps, we can return templates.
        # For now, let's assume we call the LLM.
        
        response = await self.llm.generate(prompt, system_prompt=self.system_prompt)
        
        # Extract code from markdown
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response # Fallback if no markdown
            
        ok, reason = self._validate_generated_step_code(step_id, code)
        if not ok:
            print(f"[StepGenerator] Validation failed for {step_id}: {reason}")
            # In a real app, we might retry or repair automatically
            
        return code
