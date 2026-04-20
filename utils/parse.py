import re

def extract_between(text, pre, suf):
    mt = re.search(f"{re.escape(pre)}(.*?){re.escape(suf)}", text, re.DOTALL)
    if mt:
        return mt.group(1)
    raise ValueError(f"{pre} {suf} not found in the string.")

def extract_after(text, pre):
    mt = re.search(f"{re.escape(pre)}(.*)", text, re.DOTALL)
    if mt:
        return mt.group(1)
    raise ValueError(f"{pre} found in the string.")

def extract_all_between(text, pre, suf):
    mt = re.search(f"{re.escape(pre)}(.*?){re.escape(suf)}", text)
    if not mt: return []
    return list(mt.groups())
