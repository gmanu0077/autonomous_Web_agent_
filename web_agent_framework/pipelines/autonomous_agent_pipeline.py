from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import os
import asyncio
from langgraph.graph import StateGraph, END

from ..config import TEMP_ROOT, MAX_ATTEMPTS
from ..core.utils import ensure_dir, write_text, extract_url_from_task, infer_scrape_intent
from ..core.html_context import chunk_and_pickle_html, build_html_context_from_pickles
from ..adapters.pydoll_adapter import PydollBrowserAdapter
from ..adapters.llm_factory import get_llm_adapter
from ..adapters.rag import ChromaOllamaRAG
from ..agent.planner import Planner
from ..agent.step_generator import StepGenerator
from ..agent.runner import Runner

class AgentState(TypedDict):
    task: str
    url: Optional[str]
    intent: Dict[str, bool]
    job_id: str
    job_dir: str
    attempt: int
    html_dir: str
    html_context: str
    plan: Dict[str, Any]
    shared: Dict[str, Any]
    user_demand: Optional[str]
    history: List[Dict[str, Any]]
    status: str # "init", "bootstrapped", "ready", "executing", "completed", "error"
    error: Optional[str]

# --- Nodes ---

async def bootstrap_node(state: AgentState) -> AgentState:
    print(f"[Node] Bootstrap: {state['url']}")
    # 1) Start browser
    browser = PydollBrowserAdapter()
    page = await browser.start()
    
    # 2) Navigate and scroll (Simplified bootstrap)
    await page.navigate(state["url"])
    await asyncio.sleep(5) # Initial wait
    
    # 3) Capture HTML
    html = await page.get_html()
    ensure_dir(state["html_dir"])
    chunk_and_pickle_html(html, state["html_dir"])
    
    # 4) Analysis for blockers (In a real app, this would be a step in the graph)
    # For now, we assume bootstrap is successful.
    
    await browser.close()
    
    html_context = build_html_context_from_pickles(state["html_dir"])
    
    return {
        **state,
        "status": "bootstrapped",
        "html_context": html_context
    }

async def plan_node(state: AgentState) -> AgentState:
    print("[Node] Planning")
    # Initialize Planner
    llm = get_llm_adapter()
    rag = ChromaOllamaRAG()
    planner = Planner(llm, rag)
    
    plan = await planner.plan_steps(state["task"], state["html_context"], state["job_dir"])
    
    return {
        **state,
        "plan": plan,
        "status": "planned"
    }

async def execution_node(state: AgentState) -> AgentState:
    print("[Node] Execution")
    # Execute the plan steps
    browser = PydollBrowserAdapter()
    page = await browser.start()
    
    runner = Runner(state["job_dir"])
    llm = get_llm_adapter()
    rag = ChromaOllamaRAG()
    step_gen = StepGenerator(llm)
    planner = Planner(llm, rag)
    
    step_contexts = await planner.build_step_contexts(state["plan"], state["task"], state["html_context"], state["job_dir"])
    
    step_ids = [s["id"] for s in state["plan"].get("steps", [])]
    
    for sid in step_ids:
        goal = next(s["goal"] for s in state["plan"]["steps"] if s["id"] == sid)
        context = step_contexts.get(sid, "")
        
        # Generate and run step
        code = await step_gen.generate_step(sid, goal, context, state["shared"])
        runner.save_step_module(sid, code)
        
        result = await runner.run_in_process(page, [sid], state["shared"])
        if not result["ok"]:
            print(f"[Node] Execution failed at step {sid}")
            # Decision: retry or stop?
            break
            
    await browser.close()
    
    return {
        **state,
        "status": "ready"
    }

# --- Graph Construction ---

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("bootstrap", bootstrap_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execution", execution_node)
    
    workflow.set_entry_point("bootstrap")
    workflow.add_edge("bootstrap", "plan")
    workflow.add_edge("plan", "execution")
    workflow.add_edge("execution", END)
    
    return workflow.compile()
