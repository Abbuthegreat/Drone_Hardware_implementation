#!/usr/bin/env bash
# ADAPTED from slam_kit_3/slam_kit/setup_orbslam3.sh (attached kit). Differences: it ALSO builds slam/rgbd_stream.cc (online runner used by the
# simulation) next to the kit's rgbd_offline runner. Run on Linux / Jetson. NOT executed in the Windows validation environment.
# Build Pangolin + ORB-SLAM3 + our offline RGB-D runner on a Jetson (JetPack 5/6, Ubuntu 20.04/22.04).
# Usage:   bash setup_orbslam3.sh            (installs into ~/slam_deps)
# Takes ~30-50 min on an Orin Nano. Make sure you have swap (see README).
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../slam_kit_3/slam_kit" && pwd)"
STREAM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rgbd_stream.cc"
DEPS="${DEPS:-$HOME/slam_deps}"
PANGOLIN_TAG="${PANGOLIN_TAG:-v0.8}"
# Parallel jobs: keep low on 8GB boards or the compiler gets OOM-killed
JOBS="${JOBS:-$(( $(nproc) > 4 ? 4 : $(nproc) ))}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "==> Checking apt dependencies"
PKGS="build-essential cmake git pkg-config libeigen3-dev libglew-dev libepoxy-dev libgl1-mesa-dev libboost-all-dev libssl-dev python3-pip python3-numpy libopencv-dev"
MISSING=""; for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
if [ -n "$MISSING" ]; then
    echo "Installing missing apt packages:$MISSING"
    $SUDO apt-get update && $SUDO apt-get install -y $MISSING
fi
echo "OpenCV version: $(pkg-config --modversion opencv4)"

mkdir -p "$DEPS"

# ---------------- Pangolin ----------------
if [ ! -f /usr/local/lib/cmake/Pangolin/PangolinConfig.cmake ] && [ ! -f /usr/local/lib/cmake/Pangolin/PangolinTargets.cmake ]; then
    echo "==> Building Pangolin $PANGOLIN_TAG"
    cd "$DEPS"
    [ -d Pangolin ] || git clone --depth 1 --branch "$PANGOLIN_TAG" https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF \
          -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
          -DCMAKE_CXX_FLAGS="-include cstdint"
    cmake --build build -j"$JOBS"
    $SUDO cmake --install build
    $SUDO ldconfig
else
    echo "==> Pangolin already installed"
fi

# ---------------- ORB-SLAM3 ----------------
cd "$DEPS"
[ -d ORB_SLAM3 ] || git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3

echo "==> Patching ORB-SLAM3 for newer compilers / OpenCV"
# C++14 (needed with Pangolin >= 0.7 and GCC >= 9)
sed -i 's/-std=c++11/-std=c++14/g' CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt || true
# Accept any OpenCV 4.x (JetPack 5 = 4.5, JetPack 6 = 4.8)
sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4 REQUIRED)/' CMakeLists.txt
# Old-GCC allocator issue (already fixed upstream on newer checkouts; harmless if absent)
sed -i 's/std::pair<const KeyFrame\*, g2o::Sim3>/std::pair<KeyFrame* const, g2o::Sim3>/' include/LoopClosing.h || true

# Add our runner to the build
cp "$KIT_DIR/slam/rgbd_offline.cc" Examples/RGB-D/rgbd_offline.cc
cp "$STREAM_SRC" Examples/RGB-D/rgbd_stream.cc
if ! grep -q "rgbd_offline" CMakeLists.txt; then
cat >> CMakeLists.txt <<'EOF'

# ---- added by slam_kit ----
add_executable(rgbd_offline Examples/RGB-D/rgbd_offline.cc)
target_link_libraries(rgbd_offline ${PROJECT_NAME})
set_target_properties(rgbd_offline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
add_executable(rgbd_stream Examples/RGB-D/rgbd_stream.cc)
target_link_libraries(rgbd_stream ${PROJECT_NAME})
set_target_properties(rgbd_stream PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
EOF
fi

# IMPORTANT: every part must use the SAME C++ standard. GCC 11 (JetPack 6) defaults to C++17;
# mixing C++17 g2o with C++14 ORB-SLAM3 breaks Eigen alignment -> "double free or corruption" crash.
STD="-DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "==> Building DBoW2"
cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release $STD
cmake --build Thirdparty/DBoW2/build -j"$JOBS"

echo "==> Building g2o"
cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release $STD
cmake --build Thirdparty/g2o/build -j"$JOBS"

# (Sophus is header-only for ORB-SLAM3 - no need to build it)

echo "==> Extracting vocabulary"
[ -f Vocabulary/ORBvoc.txt ] || tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary

echo "==> Building ORB-SLAM3 library + rgbd_offline (only the targets we need)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $STD
cmake --build build -j"$JOBS" --target ORB_SLAM3 rgbd_offline rgbd_stream

echo
echo "==> DONE"
echo "Runner:     $DEPS/ORB_SLAM3/Examples/RGB-D/rgbd_offline  (kit, offline)"
echo "Streaming:  $DEPS/ORB_SLAM3/Examples/RGB-D/rgbd_stream   (simulation, online)"
echo "Vocabulary: $DEPS/ORB_SLAM3/Vocabulary/ORBvoc.txt"
