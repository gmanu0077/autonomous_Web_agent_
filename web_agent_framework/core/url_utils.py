import re
import os
from urllib.parse import urljoin, urlsplit, urlunsplit
from typing import List, Tuple, Optional

def normalize_url(u: str, base_url: str = "") -> str:
    if not u: return ""
    
    # Handle srcset: "url1 100w, url2 200w" -> pick largest or last
    if ',' in u and (' ' in u or 'w' in u.lower() or 'x' in u.lower()):
        candidates = []
        for part in u.split(','):
            part = part.strip()
            if not part: continue
            sub_parts = part.split(' ')
            url_part = sub_parts[0]
            
            weight = 0
            if len(sub_parts) > 1:
                match = re.search(r'(\d+)[wx]', sub_parts[1].lower())
                if match: weight = int(match.group(1))
            
            w_match = re.search(r'[?&]width=(\d+)', url_part.lower())
            if w_match: weight = max(weight, int(w_match.group(1)))
            
            candidates.append((weight, url_part))
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            u = candidates[0][1]

    u = u.strip().strip('"\'')
    if u.startswith('//'): u = 'https:' + u
    if u.startswith('/'):
        if base_url:
            u = urljoin(base_url, u)
        else:
            u = 'https:' + u
    
    if 'images-amazon.com' in u or 'media-amazon.com' in u:
        u = u.replace('.._', '._').replace('._V1_.', '.')
    
    return u

def get_image_family_key(u: str) -> str:
    if not u: return ""
    try:
        from .image_promotion import strip_sizing_modifiers
        path = strip_sizing_modifiers(u)
        filename = os.path.basename(path).lower()
        core = os.path.splitext(filename)[0]
        
        if 'model' in core:
            match = re.search(r'(model\d*)', core)
            if match: return match.group(1)
            
        return path.lower()
    except:
        return u.lower()

def get_url_res_score(u: str) -> int:
    if not u: return -1000000
    score = 0
    u_lower = u.lower()
    
    junk = ['transparent', 'grey-pixel', '1x1', 'pixel.gif', 'tracking', 'spacer', 'sprite', 'loading', 'button', 'arrow', 'logo', 'badge', 'icon']
    if any(j in u_lower for j in junk): score -= 1000
    
    nums = re.findall(r'(\d{3,5})', u_lower)
    if nums:
        vals = [int(n) for n in nums if 50 < int(n) < 10000]
        if vals: score += max(vals)
        
    if 'hires' in u_lower or 'large' in u_lower: score += 500
    if 'thumb' in u_lower or 'small' in u_lower or 'mini' in u_lower: score -= 300
    
    return score
