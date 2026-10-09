# Drone SLAM + Human Finder (RealSense D435i/D455 + Jetson Orin)

**How it works:** record during the flight → after landing, ORB-SLAM3 builds the map, YOLO finds people,
and depth + camera pose places every person on the map.

```
flight:        record.py   ->  ~/flights/<date_time>/  (rgb/, depth/, orbslam.yaml, meta.json)
after landing: process.sh  ->  ORB-SLAM3 trajectory -> YOLO -> 3D human positions -> results/
```

Results in `~/flights/<run>/results/`:

| File | What it is |
|---|---|
| `report.html` | **Open this first.** Map, list of humans with coordinates and snapshots |
| `map_topdown.png` | Bird's-eye map: yellow flight path, red **H1, H2…** human markers, 1 m grid |
| `map_topdown_height.png` | Same map, coloured by height |
| `detections_video.mp4` | Camera feed with person boxes, human IDs, SLAM status and a live minimap |
| `map.ply`, `map_with_humans.ply` | 3D coloured map (+ red pillars at humans), for MeshLab / CloudCompare |
| `humans.csv` / `humans.json` | One row per unique person: x, y, z (m), times seen, confidence |
| `detections.csv` | Every single detection |

---

## Tonight's plan (~4 h)

| Time | Step |
|---|---|
| 0:00 | Copy this folder to the Jetson (e.g. `~/slam_kit`), run **step 1**, and leave the build running |
| 0:05 | While it builds, do **step 2** (Python packages) and **step 3** (test recording at the desk) |
| ~0:50 | Build done → **step 4**: process the desk test and check `report.html` |
| 1:15 | Mount the camera and do a short test flight (1–2 min) → process it |
| 2:00 | Real flight(s) |
| 3:00+ | Process, check the report, tune (see troubleshooting) |

### 0. Prepare (once)
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks            # max performance
# swap - the ORB-SLAM3 build needs it on 8 GB boards
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
cat /etc/nv_tegra_release                           # R35.x = JetPack 5, R36.x = JetPack 6
```

### 1. Build ORB-SLAM3 (≈30–50 min, unattended)
```bash
cd ~/slam_kit
bash setup_orbslam3.sh 2>&1 | tee build.log
```
If it gets killed (`c++: fatal error: Killed signal`), you ran out of RAM. Rerun with `JOBS=2 bash setup_orbslam3.sh`.
The script resumes where it stopped.

### 2. Python packages (while the build runs)
```bash
python3 -c "import pyrealsense2" && echo OK      # RealSense Python bindings
pip3 install onnxruntime                          # person detector (CPU, ~8-12 img/s on Orin)
```
**If `import pyrealsense2` fails**, rebuild librealsense with Python enabled (~20 min):
```bash
git clone --depth 1 https://github.com/IntelRealSense/librealsense && cd librealsense
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
make -j4 && sudo make install
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib:/usr/local/lib/python3/dist-packages' >> ~/.bashrc && source ~/.bashrc
```
The detector picks a backend automatically: **ultralytics+CUDA** if you already have a GPU PyTorch, otherwise
**onnxruntime**, otherwise **OpenCV DNN**. OpenCV DNN only works on JetPack 6 (OpenCV 4.8), not JetPack 5 (OpenCV 4.5).

### 3. Record
Camera on a **USB 3** port. Drone sitting **still** when you start, because the first second measures gravity.
```bash
cd ~/slam_kit
python3 record.py --preview      # desk test with a monitor (press q to stop)
```
On the drone, over SSH, run it inside `tmux` so it survives if SSH drops:
```bash
tmux new -s rec
python3 record.py                # Ctrl+C after landing.  Or from another shell: pkill -INT -f record.py
```
Check the printout: fps should be ~30 and `dropped` should stay near 0. Disk use is about 1 GB per minute.

### 4. Process after landing
```bash
bash process.sh                          # newest run in ~/flights
bash process.sh ~/flights/2026-09-30_22-10-00
VIEWER=1 bash process.sh                 # watch ORB-SLAM3 build the map (needs a monitor)
```
Rough times for a 5-minute flight on an Orin Nano: SLAM about 5 min (real-time replay), detection 5–10 min, map and video about 3 min.
Then open `results/report.html` in a browser, or copy the `results/` folder to your laptop.

---

## Flying tips for good SLAM (important)
- **Fly slowly and turn slowly.** Fast yaw is the #1 cause of ORB-SLAM3 losing tracking. Keep it under ~30°/s.
- Good light and **textured** scenes. Blank walls, sky, glass and uniform grass are bad.
- Stay within **~0.5–6 m** of surfaces. D435i depth gets noisy beyond ~4–5 m; D455 is fine to ~6–8 m.
- End the flight by passing over the start area again. This allows **loop closure**, which corrects drift.
- If tracking is lost, ORB-SLAM3 starts a new sub-map. Only the **largest** map is exported,
  so you'll see the `SLAM LOST` label in the video and humans in those parts won't get map positions.

## Camera pointing down instead of forward?
"Up" is normally detected from the IMU at start. If the top-down map looks like a side view, override it:
```bash
SKIP_SLAM=1 bash process.sh ~/flights/<run> --up -z    # camera facing down
SKIP_SLAM=1 bash process.sh ~/flights/<run> --up -y    # camera facing forward (level)
```

## Tuning (`SKIP_SLAM=1 bash process.sh RUN <options>` re-runs only the fast part)
| Option | Default | Meaning |
|---|---|---|
| `--conf` | 0.45 | detection confidence. Lower it (0.3) to find more people, raise it if you get false alarms |
| `--stride` | 2 | detect every Nth frame (1 = every frame, slower) |
| `--radius` | 1.5 | detections closer than this (m) count as the same person |
| `--min-obs` | 3 | sightings needed to confirm a human. Lower it if people are only seen briefly |
| `--max-depth` | 8 | ignore people farther than this (m), because depth gets unreliable |
| `--map-depth` / `--voxel` | 5 / 0.05 | map range and resolution |

SLAM keeps losing track? Replay slower: `SPEED=0.5 bash process.sh RUN`. You can also raise `ORBextractor.nFeatures`
in the run's `orbslam.yaml` to 1500–2000.

## Troubleshooting
- **`double free or corruption` right after "New Map created"**: g2o/DBoW2 were built with a different C++ standard
  than ORB-SLAM3. v2 of `setup_orbslam3.sh` fixes this. If you built with v1, rebuild only the two small libs (≈3 min):
  ```bash
  cd ~/slam_deps/ORB_SLAM3
  S="-DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF"
  rm -rf Thirdparty/g2o/build Thirdparty/DBoW2/build
  cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release $S && cmake --build Thirdparty/DBoW2/build -j4
  cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release $S && cmake --build Thirdparty/g2o/build -j4
  ```
- **`No RealSense camera found` / low fps**: use a USB 3 port and cable, and make sure `realsense-viewer` is closed. Only one program can use the camera at a time.
- **Detector error**: run `pip3 install onnxruntime`. OpenCV DNN can't run YOLOv8 on JetPack 5's OpenCV 4.5.
- **Humans found but in odd places**: check the video. `SLAM LOST` frames can't be placed. People farther than ~6 m
  (D435i) get noisy depth, so try `--max-depth 5`.

## Faster detection (optional, later)
Install NVIDIA's PyTorch for your JetPack plus `pip3 install ultralytics`, then export a TensorRT engine:
`yolo export model=models/yolov8n.pt format=engine half=True` and run with
`--model models/yolov8n.engine --backend ultralytics`. This gives about 40+ img/s instead of about 10.

## Files
- `record.py`: RealSense recorder. It writes `orbslam.yaml` with the camera's own factory intrinsics, so no calibration is needed.
- `setup_orbslam3.sh`: builds Pangolin, ORB-SLAM3 and `slam/rgbd_offline.cc` (our ORB-SLAM3 runner).
- `process.sh`: runs SLAM, then `postprocess.py`.
- `postprocess.py`: detection, 3D localisation, clustering into unique humans, map, video and report.
- `detector.py`: YOLOv8n person detector (ultralytics / onnxruntime / OpenCV backends).
- `models/yolov8n.onnx`, `models/yolov8n.pt`: COCO-pretrained YOLOv8n (class 0 = person).
