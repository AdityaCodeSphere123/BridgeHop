#!/bin/zsh
# Wait for the daily token quota to reset (midnight IST), then run the queued experiments.
cd /Users/aditya/Desktop/classes/COL764-Assignment4
until [ "$(date +%Y%m%d%H%M)" -ge 202611060003 ]; do sleep 60; done
for v in b bev; do
  python3 dev/exp/crun.py --agent dev/exp/agent_$v.py --limit 150 --out runs/$v --workers 8 > runs/$v.log 2>&1
  echo "== $v"; grep -E "^S |answer F1|evidence F1|status|mean" runs/$v.log
done
