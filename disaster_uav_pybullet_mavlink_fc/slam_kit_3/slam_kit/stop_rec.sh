#!/usr/bin/env bash
# Stop the recording cleanly and wait until every frame is saved.
if ! pgrep -f "^python3 record.py" >/dev/null; then echo "Not recording."; exit 1; fi
pkill -INT -f "^python3 record.py"
echo "Stopping... saving remaining frames"
while pgrep -f "^python3 record.py" >/dev/null; do sleep 1; done
tail -3 ~/rec.log
echo ">>> Saved. Now process it with: bash ~/slam_kit/process.sh"
