def js_get_smart_dom_tree() -> str:
    return r"""
    (function(){
      const results = [];
      let nextId = 0;
      const scrollY = window.scrollY || window.pageYOffset;
      const scrollX = window.scrollX || window.pageXOffset;
      
      function safeStringify(obj) {
        try {
          const cache = new Set();
          const str = JSON.stringify(obj, (key, value) => {
            if (typeof value === 'object' && value !== null) {
              if (cache.has(value)) return;
              cache.add(value);
            }
            return value;
          });
          return str;
        } catch (e) { return null; }
      }

      function walk(el, parentId = null, depth = 0) {
        if (!el || el.nodeType !== 1 || depth > 40) return;
        const tag = el.tagName.toLowerCase();
        if (['style', 'noscript', 'svg', 'path', 'head', 'meta', 'link', 'iframe'].includes(tag)) return;
        
        if (tag === 'script') {
          const content = el.textContent || "";
          if (content.length > 50 && (content.includes('Image') || content.includes('jpg') || content.includes('png') || content.includes('gallery'))) {
            results.push({
              id: "node_" + (nextId++), parentId: parentId, tag: tag, depth: depth,
              text: content.slice(0, 100000), attrs: { type: el.type || "" },
              rect: { top: 0, left: 0, width: 0, height: 0, area: 0 }
            });
          }
          return;
        }

        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const isHidden = style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || rect.width === 0 || rect.height === 0;
        const isImportant = el.id.toLowerCase().includes('image') || el.id.toLowerCase().includes('gallery') || tag === 'img';
        
        if (isHidden && !isImportant && depth > 15) return;

        const attrs = {};
        for (let i = 0; i < el.attributes.length; i++) {
          const attr = el.attributes[i];
          attrs[attr.name] = attr.value;
        }

        let jqueryData = null;
        try {
          if (window.jQuery) {
            const data = jQuery(el).data();
            if (data && Object.keys(data).length > 0) {
              const filtered = {};
              for (const k in data) {
                if (k.toLowerCase().includes('image') || k.toLowerCase().includes('gallery') || k.toLowerCase().includes('data')) {
                  filtered[k] = data[k];
                }
              }
              if (Object.keys(filtered).length > 0) jqueryData = safeStringify(filtered);
            }
          }
        } catch (e) {}

        const bgImg = style.backgroundImage;
        let bgUrl = null;
        if (bgImg && bgImg !== 'none' && bgImg.includes('url')) {
          const match = bgImg.match(/url\("?(.+?)"?\)/);
          if (match) bgUrl = match[1];
        }

        let directText = "";
        for (let i = 0; i < el.childNodes.length; i++) {
          if (el.childNodes[i].nodeType === 3) directText += el.childNodes[i].textContent;
        }
        directText = directText.trim().replace(/\s+/g, ' ');

        const currentId = "node_" + (nextId++);
        const node = {
          id: currentId, parentId: parentId, tag: tag, depth: depth, attrs: attrs,
          text: directText.slice(0, 1000), bgUrl: bgUrl, jqueryData: jqueryData,
          rect: {
            top: Math.round(rect.top + scrollY), left: Math.round(rect.left + scrollX),
            width: Math.round(rect.width), height: Math.round(rect.height),
            area: Math.round(rect.width * rect.height)
          },
          computed: { zIndex: style.zIndex, opacity: style.opacity, position: style.position, display: style.display, visibility: style.visibility }
        };

        if (tag === 'img') {
          node.src = el.src;
          node.data_src = el.getAttribute('data-src') || el.getAttribute('data-lazy-src') || el.getAttribute('data-zoom-image');
          node.srcset = el.srcset || el.getAttribute('data-srcset');
          node.alt = el.alt;
        } else if (tag === 'a') {
          node.href = el.href;
        }

        results.push(node);
        for (let i = 0; i < el.children.length; i++) walk(el.children[i], currentId, depth + 1);
      }
      walk(document.body);
      return JSON.stringify(results);
    })();
    """

def js_click_cookieish() -> str:
    return r"""
    (function(){
      const words = ['accept','agree','ok','okay','got it','continue','allow','yes','i understand'];
      const closeWords = ['close','dismiss','hide','decline','refuse','no thanks','skip','×','✕','✕','✕','✖','×'];
      const nodes = Array.from(document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"], a.close, button.close, [class*="close"], [id*="close"], [aria-label*="close"], [title*="close"]'));
      let clicked = 0;
      for(const el of nodes){
        const text = (el.textContent||'').trim().toLowerCase();
        const combined = text + ' ' + (el.getAttribute('aria-label')||'').toLowerCase() + ' ' + (el.getAttribute('title')||'').toLowerCase();
        let s = 0;
        for(const w of words){ if(combined.includes(w)) s += 2; }
        for(const w of closeWords){ if(combined.includes(w)) s += 3; }
        if(s > 0 && el.offsetWidth > 0 && el.offsetHeight > 0){
          try { el.click(); clicked++; } catch(e) {}
        }
      }
      return JSON.stringify({clicked});
    })();
    """

def js_expand_pdp_sections() -> str:
    """Alias for js_expand_sections_universal (backward compatibility)."""
    return js_expand_sections_universal()


def js_expand_sections_universal() -> str:
    """Universal section expansion. No site-specific selectors — works across the whole web."""
    return r"""
    (function(){
      const KEYWORDS = [
        'more','show more','see more','view more','load more','expand','read more',
        'description','details','about',
        'comments','load comments','view comments','show comments',
        'related','recommendations','suggested'
      ];
      function norm(s){ return (s||'').toString().replace(/\s+/g,' ').trim().toLowerCase(); }
      const candidates = Array.from(document.querySelectorAll(
        'button, [role="button"], summary, [aria-expanded="false"], [data-testid*="accordion"], [data-testid*="expand"], [data-testid*="details"], [class*="expand"], [class*="more"], [class*="load-more"]'
      ));
      let clicked = 0;
      for(const el of candidates){
        const t = norm(el.textContent) + ' ' + norm(el.getAttribute('aria-label')) + ' ' + norm(el.getAttribute('title')) + ' ' + norm(el.className);
        if(KEYWORDS.some(k => t.includes(k))){
          try { el.scrollIntoView({block:'center'}); el.click(); clicked++; } catch(e) {}
        }
      }
      return JSON.stringify({clicked});
    })();
    """


def js_universal_scroll() -> str:
    """Scroll down the page in steps to trigger lazy loading (comments, related, infinite scroll)."""
    return r"""
    (function(){
      const step = 600;
      const pause = 400;
      const maxScrolls = 15;
      const docHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
      const viewHeight = window.innerHeight;
      const maxY = Math.max(0, docHeight - viewHeight);
      function scrollStep(i) {
        if (i >= maxScrolls) {
          window.scrollTo(0, 0);
          return Promise.resolve(JSON.stringify({scrolls: i, docHeight}));
        }
        const targetY = Math.min((i + 1) * step, maxY);
        window.scrollTo(0, targetY);
        return new Promise(r => setTimeout(r, pause)).then(() => scrollStep(i + 1));
      }
      return scrollStep(0);
    })();
    """

def js_probe_image_meta() -> str:
    return r"""
    (urls) => {
        const probeOne = (url) => {
            return new Promise((resolve) => {
                const img = new Image();
                img.crossOrigin = "anonymous";
                const t = setTimeout(() => { img.src = ""; resolve({ u: url, error: "timeout" }); }, 10000);
                img.onload = () => {
                    clearTimeout(t);
                    resolve({ u: url, w: img.naturalWidth, h: img.naturalHeight, a: (img.naturalWidth * img.naturalHeight) || 0 });
                };
                img.onerror = () => { clearTimeout(t); resolve({ u: url, error: "load_failed" }); };
                img.src = url;
            });
        };
        return Promise.all(urls.map(u => probeOne(u))).then(results => JSON.stringify(results));
    }
    """
