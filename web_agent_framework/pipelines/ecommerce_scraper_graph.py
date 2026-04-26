import asyncio
import json
import os
import re
from typing import List, Dict, Any, TypedDict, Optional
import networkx as nx
from langgraph.graph import StateGraph, END

from ..adapters.browser import BrowserAdapter, PageAdapter
from ..adapters.llm import LLMAdapter
from ..adapters.pydoll_adapter import PydollBrowserAdapter
from ..adapters.gemini_adapter import GeminiAdapter
from ..core.url_utils import normalize_url, get_image_family_key, get_url_res_score
from ..core.image_promotion import speculative_promote_url, strip_sizing_modifiers
from ..core.dom_graph import DOMGraphBuilder, perform_smart_chunking
from ..core.browser_js_snippets import js_get_smart_dom_tree, js_click_cookieish, js_expand_pdp_sections, js_probe_image_meta
from ..config import GEMINI_API_KEY

class GraphState(TypedDict):
    url: str
    html: Optional[str]
    graph: Optional[nx.DiGraph]
    chunks: List[Dict[str, Any]]
    product_data: Dict[str, Any]
    main_images: List[Dict[str, Any]]
    errors: List[str]
    metadata: Dict[str, Any]

class EcommerceGraphScraper:
    def __init__(self, browser_adapter: BrowserAdapter, llm_adapter: LLMAdapter):
        self.browser = browser_adapter
        self.llm = llm_adapter

    async def fetch_html_node(self, state: GraphState) -> GraphState:
        print(f"[*] Fetching HTML for {state['url']}...")
        try:
            page = await self.browser.start()
            await page.navigate(state['url'])
            
            # Stealth/Wait logic
            await asyncio.sleep(8)
            
            # Interactions
            await page.execute_script(js_click_cookieish())
            await asyncio.sleep(1)
            await page.execute_script("window.scrollTo(0, 500);")
            await asyncio.sleep(1)
            await page.execute_script(js_expand_pdp_sections())
            await asyncio.sleep(1)
            
            # Click thumbnails (simplified from graph_scraper)
            click_thumb_js = """
            (function(){
                const selectors = ['li[class*="thumb"]', 'img[class*="thumb"]', '.alt-image', '.product-image-thumb'];
                selectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach((t, i) => {
                        if (i < 5) { try { t.click(); } catch(e){} }
                    });
                });
            })();
            """
            await page.execute_script(click_thumb_js)
            await asyncio.sleep(2)
            
            # Capture DOM nodes
            dom_nodes = await page.execute_script(js_get_smart_dom_tree())
            if isinstance(dom_nodes, str):
                dom_nodes = json.loads(dom_nodes)
            state['metadata']['dom_nodes_raw'] = dom_nodes
            
            html = await page.get_html()
            state['html'] = html
            
            await self.browser.close()
        except Exception as e:
            state['errors'].append(f"Fetch error: {str(e)}")
        return state

    async def build_graph_node(self, state: GraphState) -> GraphState:
        print("[*] Building DOM graph...")
        dom_nodes = state['metadata'].get('dom_nodes_raw', [])
        if dom_nodes:
            G = nx.DiGraph()
            for node in dom_nodes:
                node_id = node['id']
                parent_id = node.get('parentId')
                props = {k: v for k, v in node.items() if k not in ['id', 'parentId']}
                G.add_node(node_id, **props)
                if parent_id:
                    G.add_edge(parent_id, node_id)
            state['graph'] = G
        elif state['html']:
            builder = DOMGraphBuilder()
            state['graph'] = builder.build(state['html'])
        return state

    async def chunk_graph_node(self, state: GraphState) -> GraphState:
        print("[*] Chunking graph...")
        if state['graph']:
            state['chunks'] = perform_smart_chunking(state['graph'], state['url'])
        return state

    async def extract_product_node(self, state: GraphState) -> GraphState:
        print("[*] Extracting product details...")
        if not state['chunks']: return state
        
        pdp_chunks = [c for c in state['chunks'] if any(k in c['text'].lower() for k in ['price', 'add to cart', 'description'])]
        pdp_chunks.sort(key=lambda x: len(x['text']), reverse=True)
        
        context = ""
        for i, c in enumerate(pdp_chunks[:10]):
            context += f"\nCHUNK {i}: {c['text'][:1000]}\n"

        prompt = f"Extract product Title, Brand, Price, Currency, Description, Features as JSON from these chunks for {state['url']}:\n{context}"
        
        try:
            state['product_data'] = await self.llm.generate_json(prompt)
        except Exception as e:
            state['errors'].append(f"Product extraction error: {str(e)}")
        return state

    async def extract_images_node(self, state: GraphState) -> GraphState:
        print("[*] Extracting images...")
        # Simplified image selection for modular version
        # In a full implementation, we'd use the gallery identification logic from graph_scraper
        all_candidate_urls = []
        for c in state['chunks']:
            for img in c.get('images', []):
                src = img.get('src') or img.get('data_src')
                if src:
                    all_candidate_urls.append(normalize_url(src, state['url']))
        
        # Deduplicate and promote
        unique_images = {}
        for u in all_candidate_urls:
            fk = get_image_family_key(u)
            score = get_url_res_score(u)
            if fk not in unique_images or score > unique_images[fk][0]:
                unique_images[fk] = (score, u)
        
        state['main_images'] = [{"url": url, "score": score} for score, url in unique_images.values()]
        state['main_images'].sort(key=lambda x: x['score'], reverse=True)
        
        return state

    def create_workflow(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("fetch", self.fetch_html_node)
        workflow.add_node("build", self.build_graph_node)
        workflow.add_node("chunk", self.chunk_graph_node)
        workflow.add_node("details", self.extract_product_node)
        workflow.add_node("images", self.extract_images_node)
        
        workflow.set_entry_point("fetch")
        workflow.add_edge("fetch", "build")
        workflow.add_edge("build", "chunk")
        workflow.add_edge("chunk", "details")
        workflow.add_edge("details", "images")
        workflow.add_edge("images", END)
        
        return workflow.compile()

async def run_ecommerce_graph_pipeline(url: str):
    browser = PydollBrowserAdapter()
    # Assuming we use Gemini for this specific pipeline as it was the original
    llm = GeminiAdapter(api_key=GEMINI_API_KEY)
    
    scraper = EcommerceGraphScraper(browser, llm)
    app = scraper.create_workflow()
    
    state = {
        "url": url,
        "html": None,
        "graph": None,
        "chunks": [],
        "product_data": {},
        "main_images": [],
        "errors": [],
        "metadata": {}
    }
    
    return await app.ainvoke(state)
