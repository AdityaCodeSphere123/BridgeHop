"""run_dev.py with a disk cache for LLM calls (temperature 0, so identical prompts give identical outputs).

    python dev/exp/crun.py --agent dev/exp/agent_x.py --out runs/x      # same flags as run_dev.py
"""
import json, os, sys, threading, atexit, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import a4.llm as L
import run_dev

CACHE_PATH = os.path.join(ROOT, "runs", "llm_cache.jsonl")
_lock = threading.Lock()
_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        for line in f:
            try:
                r = json.loads(line); _cache[r["k"]] = r["v"]
            except Exception:
                pass
_fh = open(CACHE_PATH, "a")

def _key(*parts):
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()

_orig_complete = L.HTTPLLM.complete
_orig_count = L.HTTPLLM.count_tokens

def complete(self, prompt, max_tokens):
    k = _key("c", prompt, max_tokens)
    with _lock:
        if k in _cache:
            return tuple(_cache[k])
    v = _orig_complete(self, prompt, max_tokens)
    with _lock:
        _cache[k] = list(v); _fh.write(json.dumps({"k": k, "v": list(v)}) + "\n"); _fh.flush()
    return v

def count_tokens(self, prompt):
    k = _key("t", prompt)
    with _lock:
        if k in _cache:
            return _cache[k]
    v = _orig_count(self, prompt)
    with _lock:
        _cache[k] = v; _fh.write(json.dumps({"k": k, "v": v}) + "\n"); _fh.flush()
    return v

L.HTTPLLM.complete = complete
L.HTTPLLM.count_tokens = count_tokens
run_dev.HTTPLLM = L.HTTPLLM
if __name__ == "__main__":
    run_dev.main()
