#!/bin/zsh
cd /Users/aditya/Desktop/classes/COL764-Assignment4
target=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-11-16 00:02:00" +%s); now=$(date +%s)
(( target > now )) && sleep $(( target - now ))
python3 dev/exp/crun.py --agent dev/exp/r11/agent_clean.py --out runs/r11_clean > dev/exp/r11/log_clean.txt 2>&1
python3 dev/exp/crun.py --agent dev/exp/r11/agent_cleanB.py --out runs/r11_cleanB > dev/exp/r11/log_cleanB.txt 2>&1
grep -h -A8 "results" dev/exp/r11/log_clean.txt dev/exp/r11/log_cleanB.txt
