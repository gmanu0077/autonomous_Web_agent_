import re
from typing import List

def strip_sizing_modifiers(u: str) -> str:
    if not u: return u
    if '/images/I/' in u or '/images/S/' in u or '/images/G/' in u:
        u = re.sub(r'\.[_][^/]+(?=\.[a-z]{3,4}$)', '', u, flags=re.I)
        u = re.sub(r'\.(?:AC|SR|SX|SY|SL|UL)\d+[^/]*?(?=\.[a-z]{3,4}$)', '', u, flags=re.I)
    
    u = re.sub(r"(?i)[-_.]\d{1,5}[xXwW]\d{0,5}[hH]?(?=[._-]|$)", "", u)
    u = re.sub(r"(?i)[-_.] (?:thumb|small|medium|large|grande|icon|mini|v[0-9]+)(?=[._-]|$)", "", u)
    
    if '?' in u:
        base, query = u.split('?', 1)
        params = query.split('&')
        filtered = [p for p in params if not any(k in p.lower() for k in ['width=', 'height=', 'w=', 'h=', 'size='])]
        u = f"{base}?{'&'.join(filtered)}" if filtered else base
            
    return u

def speculative_promote_url(u: str) -> List[str]:
    candidates = []
    clean = strip_sizing_modifiers(u)
    if clean != u: candidates.append(clean)
    
    hr_targets = ['1500', '2000', '2500']
    if '/images/I/' in u or '/images/S/' in u or '/images/G/' in u:
        ext_match = re.search(r'(\.[a-z]{2,4})$', clean, flags=re.I)
        ext = ext_match.group(1) if ext_match else ".jpg"
        base = clean[:ext_match.start()] if ext_match else clean
        for target in hr_targets:
            candidates.extend([f"{base}._SL{target}_{ext}", f"{base}._SX{target}_{ext}", f"{base}._SY{target}_{ext}"])

    dim_match = re.search(r'([-_.]\d{1,5}[xXwW]\d{0,5}[hH]?(?:[xX]\d{1,5}[hH]?)?)', u)
    if dim_match:
        orig = dim_match.group(1)
        sep = 'W' if 'w' in orig.lower() else 'x'
        has_h = 'h' in orig.lower()
        for target in ['1000', '1200', '1500', '2000']:
            new_b = f"{orig[0]}{target}{sep}{target}"
            if has_h: new_b += 'H'
            candidates.append(u.replace(orig, new_b))

    if 'width=' in u.lower() or 'w=' in u.lower():
        for target in ['1600', '2000', '2400', '3000', '4000']:
            new_u = re.sub(r'([?&](?:width|w|width_))(\d+)', rf'\1{target}', u, flags=re.I)
            if new_u != u: candidates.append(new_u)
            
    quality_map = {'thumb': 'large', 'small': 'hires', 'mini': 'original', 'medium': 'original'}
    for low, high in quality_map.items():
        if low in u.lower(): candidates.append(u.lower().replace(low, high))
            
    return list(set(candidates))
