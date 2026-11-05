exec(open("dev/exp/hop2.py").read().split('for name in S:')[0])
def rrf(lists, c=3):
    sc = {}
    for L in lists:
        for i, d in enumerate(L): sc[d] = sc.get(d, 0) + 1 / (c + i + 1)
    return sorted(sc, key=lambda d: -sc[d])
def rr(lists):
    out = []
    for i in range(10):
        for L in lists:
            if i < len(L) and L[i] not in out: out.append(L[i])
    return out
for combo in [("q2b", "b+q"), ("q2b", "b+q", "b"), ("q2b", "b"), ("b+q2b", "b+q", "b"), ("q2b","b+q2b","b+q")]:
    for fn in (rrf, rr):
        for N_ in (8, 10, 12, 15, 20):
            hit = sum(r["h2"] in fn([res(n, r, 10) for n in combo])[:N_] for r in rows) / len(rows)
            print(f"{'+'.join(combo):18s} {fn.__name__:4s} top{N_:2d} {hit:.3f}")
