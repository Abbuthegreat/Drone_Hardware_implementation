#!/usr/bin/env bash
# After landing:  bash process.sh ~/flights/<run>          (or no argument = newest run)
# Env options:    VIEWER=1   show ORB-SLAM3 viewer (needs a monitor)
#                 SPEED=0.5  SLAM playback speed (1 = real time; lower = more robust, slower)
#                 SKIP_SLAM=1  re-run only the detection/map step
#                 any extra args are passed to postprocess.py (e.g. --up -z --conf 0.35)
set -euo pipefail
if pgrep -f "^python3 record.py" >/dev/null; then
    echo "Recording is still running - stop it first:  bash ~/slam_kit/stop_rec.sh"; exit 1
fi
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB="${ORB_SLAM3_DIR:-$HOME/slam_deps/ORB_SLAM3}"

RUN="${1:-}"
if [ -z "$RUN" ] || [[ "$RUN" == --* ]]; then
    RUN="$(ls -d "$HOME"/flights/*/ 2>/dev/null | sort | tail -1)"
    [ -z "$RUN" ] && { echo "No run folder given and none in ~/flights"; exit 1; }
else
    shift
fi
RUN="$(cd "$RUN" && pwd)"
echo "==> Run: $RUN"

[ -s "$RUN/associations.txt" ] || python3 "$KIT/record.py" --repair "$RUN"

if [ "${SKIP_SLAM:-0}" != "1" ]; then
    echo "==> ORB-SLAM3 (RGB-D) ... loading the vocabulary takes ~30-60 s"
    "$ORB/Examples/RGB-D/rgbd_offline" "$ORB/Vocabulary/ORBvoc.txt" "$RUN/orbslam.yaml" "$RUN" \
        "${VIEWER:-0}" "${SPEED:-1.0}" 2>&1 | tee "$RUN/slam.log" || true
    [ -s "$RUN/CameraTrajectory.txt" ] || echo "!! SLAM produced no trajectory - continuing with detection only"
fi

echo "==> Detection + human localisation + map"
python3 "$KIT/postprocess.py" "$RUN" "$@"
echo
echo "Open:  $RUN/results/report.html"
