#!/usr/bin/env bash
# Start recording in the background (survives SSH disconnects). Stop with: bash ~/slam_kit/stop_rec.sh
if pgrep -f "^python3 record.py" >/dev/null; then echo "Already recording!"; tail -2 ~/rec.log; exit 1; fi
tmux kill-session -t rec 2>/dev/null
tmux new -d -s rec ". ~/slam_venv/bin/activate && cd ~/slam_kit && python3 record.py $* 2>&1 | tee ~/rec.log"
sleep 8
tail -4 ~/rec.log
pgrep -f "^python3 record.py" >/dev/null && echo ">>> RECORDING. Stop with: bash ~/slam_kit/stop_rec.sh" || echo ">>> FAILED to start - see above"
