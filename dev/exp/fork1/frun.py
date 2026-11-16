"""crun.py plus injected server failures: a fixed RATE of LLM calls (chosen by prompt hash) raise LLMError
after being counted, like a server error after the client's retries.
    FAIL_RATE=0.05 python3 dev/exp/fork1/frun.py --agent X --out runs/Y"""
import os, sys, hashlib
ROOT = "/Users/aditya/Desktop/classes/COL764-Assignment4"
sys.path.insert(0, ROOT + "/dev/exp"); sys.argv[0] = ROOT + "/dev/exp/crun.py"
import crun
import a4.llm as L
from a4.budget import LLMError
RATE = float(os.environ.get("FAIL_RATE", "0.05"))
_cached = L.HTTPLLM.complete
def complete(self, prompt, max_tokens):
    if int(hashlib.sha1(("fail" + prompt).encode()).hexdigest(), 16) % 10000 < RATE * 10000:
        raise LLMError("injected server failure")
    return _cached(self, prompt, max_tokens)
L.HTTPLLM.complete = complete
crun.run_dev.HTTPLLM = L.HTTPLLM
if __name__ == "__main__":
    crun.run_dev.main()
