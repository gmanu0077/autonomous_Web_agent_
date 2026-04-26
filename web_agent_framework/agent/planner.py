import os
import json
import re
from typing import Dict, Any, Tuple, List
from ..config import (
    MAX_CONTEXT_CHARS,
    MAX_HTML_CONTEXT_CHARS,
    TOP_K,
    PLANNER_MODEL,
)
from ..core.utils import write_text
from ..core.html_context import build_html_context_from_pickles, build_dom_js_hints
from ..adapters.rag import ChromaOllamaRAG
from ..adapters.llm import LLMAdapter

# Allowed step IDs for the autonomous agent
ALLOWED_STEP_IDS = [
    "page_analysis",
    "handle_blockers",
    "collect_dom_imgs_basic",
    "positional_grouping",
    "cluster_and_filter",
    "custom_operation", # Added for arbitrary user demands
]

REQUIRED_ORDER = [
    "page_analysis",
    "handle_blockers",
    "collect_dom_imgs_basic",
    "positional_grouping",
    "cluster_and_filter",
]

class Planner:
    def __init__(self, llm_adapter: LLMAdapter, rag_adapter: ChromaOllamaRAG):
        self.llm = llm_adapter
        self.rag = rag_adapter

    def default_plan(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "id": "page_analysis",
                    "goal": "OFFLINE-FIRST: read pickle HTML chunks and build shared['page_analysis'] JSON (layout, overlays, media, product fields).",
                },
                {
                    "id": "handle_blockers",
                    "goal": "Dismiss cookie banners / region modals / newsletter overlays using page_analysis info.",
                },
                {
                    "id": "collect_dom_imgs_basic",
                    "goal": "Harvest visible DOM images + bounding boxes via execute_script and fill shared['candidate_urls'] + shared['url_meta'].",
                },
                {
                    "id": "positional_grouping",
                    "goal": "Using rect data, assign region hints (header/main/footer/sidebar).",
                },
                {
                    "id": "cluster_and_filter",
                    "goal": "Score/filter candidate URLs to isolate the main product gallery.",
                },
            ]
        }

    async def plan_steps(self, task_description: str, html_context: str, attempt_dir: str) -> Dict[str, Any]:
        print("[Planner] Generating step plan...")
        html_snippet = (html_context or "")[:1600]

        prompt = f"""
You are a senior scraping architect.
Produce a JSON plan of 4-6 ordered steps for the following task.
Use ONLY these step IDs: {ALLOWED_STEP_IDS}

Step semantics:
- page_analysis: offline parse of HTML chunks into shared["page_analysis"]
- handle_blockers: close overlays
- collect_dom_imgs_basic: candidates from page_analysis or DOM scan
- positional_grouping: region hints using rect data
- cluster_and_filter: pick final results
- custom_operation: specific user demand not covered by standard steps

TASK: {task_description}
HTML_SNIPPET: {html_snippet}

Return ONLY valid JSON:
{{
  "steps": [
    {{"id": "...", "goal": "..."}},
    ...
  ]
}}
"""
        try:
            plan = await self.llm.generate_json(prompt, system_prompt="You output only minimal JSON plans for scraping steps.")
        except Exception as e:
            print(f"[Planner] LLM failed: {e}")
            plan = self.default_plan()

        # Validation and normalization
        if not plan or "steps" not in plan:
            plan = self.default_plan()
        else:
            seen = set()
            cleaned_steps = []
            for st in plan["steps"]:
                sid = st.get("id")
                if sid in ALLOWED_STEP_IDS and sid not in seen:
                    seen.add(sid)
                    cleaned_steps.append({"id": sid, "goal": st.get("goal", "")})
            
            # Ensure required order for standard steps if present
            order_map = {sid: i for i, sid in enumerate(REQUIRED_ORDER)}
            cleaned_steps.sort(key=lambda s: order_map.get(s["id"], 999))
            plan = {"steps": cleaned_steps}

        write_text(os.path.join(attempt_dir, "plan.json"), json.dumps(plan, indent=2))
        return plan

    async def build_step_contexts(self, plan: Dict[str, Any], task_description: str, html_context: str, attempt_dir: str) -> Dict[str, str]:
        step_contexts = {}
        steps = plan.get("steps", [])
        num_steps = max(1, len(steps))
        per_step_chars = max(1000, MAX_CONTEXT_CHARS // num_steps)
        html_snippet = (html_context or "")[:800]

        for step in steps:
            sid = step["id"]
            goal = step["goal"]
            query = f"{task_description}\nSTEP_ID: {sid}\nGOAL: {goal}\nHTML: {html_snippet}"
            
            chunks = self.rag.get_top_chunks(query, n_results=TOP_K)
            ctx = self.rag.build_context_from_chunks(chunks, max_chars=per_step_chars) if chunks else "(No context)"
            
            step_contexts[sid] = ctx
            write_text(os.path.join(attempt_dir, f"step_context_{sid}.txt"), ctx)
        
        return step_contexts
