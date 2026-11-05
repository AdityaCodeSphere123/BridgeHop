exec(open("dev/exp/fuse.py").read().split('for combo in')[0])
S["q1"] = lambda r: r["q1"]; S["q"] = lambda r: r["q"]
for c in (1, 3, 10, 60):
    def rrfc(lists, c=c):
        sc = {}
        for L in lists:
            for i, d in enumerate(L): sc[d] = sc.get(d, 0) + 1 / (c + i + 1)
        return sorted(sc, key=lambda d: -sc[d])
    print("c", c, " ".join(f"top{n}={sum(r['h1'] in rrfc([res('q1', r, 10), res('q', r, 10)])[:n] for r in rows)/len(rows):.3f}" for n in (5, 8, 10, 12, 15, 20)))
