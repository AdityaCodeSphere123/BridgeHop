import importlib.util, os
H = os.path.dirname(os.path.abspath(__file__))
def _p(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(H, n + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.PATCH
PATCH = _p("cit_overlap1") + _p("drop_unit")
