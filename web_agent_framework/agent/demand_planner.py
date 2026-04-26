"""
Demand-aware planner for Phase 3.

Produces step plans for arbitrary user demands (e.g. "Find the price", "Add to cart")
using the pre-built DOM graph and HTML context.
"""

import json
import re
from typing import Dict, Any

from ..adapters.llm import LLMAdapter


class DemandPlanner:
    """Plans steps for arbitrary user demands using DOM graph and HTML context."""

    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def plan_for_demand(
        self,
        user_demand: str,
        html_context: str,
        dom_js_hints: str,
        graph_chunks_text: str,
    ) -> Dict[str, Any]:
        """
        Generate a step plan to fulfill the user's demand.

        Returns plan with shape: {"steps": [{"id": "...", "goal": "..."}, ...]}
        """
        prompt = f"""
You are a web automation architect. The user has loaded a page and wants you to fulfill this demand:

USER DEMAND: {user_demand}

You have access to:
1. HTML context (truncated)
2. DOM/JS hints
3. Graph chunks (semantic regions of the page)

HTML_CONTEXT (truncated):
{html_context[:6000]}

DOM_JS_HINTS:
{dom_js_hints[:1500]}

GRAPH_CHUNKS (page regions):
{graph_chunks_text[:4000]}

Produce a JSON plan of 1-5 ordered steps to fulfill the demand. Each step should:
- Have a unique id (e.g. find_price, add_to_cart, get_shipping_terms, extract_variant_info)
- Have a clear goal describing what the step does
- Use execute_script for DOM interaction (no tab.go_to, no navigation)
- Return JSON-serializable data only

For comments, related videos, or lazy-loaded content (e.g. YouTube): include a scroll step BEFORE extraction.
Example: {{"id": "scroll_comments", "goal": "Scroll the comments section to trigger lazy loading"}} then {{"id": "get_comments", "goal": "..."}}

Examples of step IDs for common demands:
- find_price, get_price, extract_price
- add_to_cart, click_add_to_cart
- get_shipping_terms, find_shipping_info
- select_variant, choose_color, pick_size
- extract_product_info, get_product_details
- scroll_comments, scroll_related_videos (for lazy-loaded sections)
- get_video_metadata, get_related_videos, get_comments (YouTube-style)

Return ONLY valid JSON:
{{
  "steps": [
    {{"id": "step_1", "goal": "..."}},
    {{"id": "step_2", "goal": "..."}}
  ]
}}
"""
        resp = ""
        try:
            resp = await self.llm.generate(prompt, system_prompt="You output only valid JSON. No markdown, no explanations.")
            clean = (resp or "").strip()
            if clean.startswith("```"):
                parts = clean.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:]
                    if p.startswith("{"):
                        return json.loads(p)
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\"steps\"[^{}]*\[.*?\][^{}]*\}", resp or "", re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
        except Exception as e:
            print(f"[DemandPlanner] LLM failed: {e}")

        # Fallback minimal plan
        return {
            "steps": [
                {"id": "extract_info", "goal": f"Extract information from the page to fulfill: {user_demand}"},
            ]
        }
