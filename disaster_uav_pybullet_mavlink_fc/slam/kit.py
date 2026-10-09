"""
Access to the attached physical SLAM kit (slam_kit_3/slam_kit) WITHOUT modifying it.

Reused as-is:   detector.Detector (YOLOv8n person detector, kit model yolov8n.onnx)
                postprocess.localize (torso-median-depth 3-D person point), postprocess.cluster,
                postprocess.rot_a_to_b (gravity alignment), postprocess.TopDown / render_topdown / VoxelMap / write_ply
                record.write_orbslam_yaml (ORB-SLAM3 settings from camera intrinsics), record.write_associations
Adapted (new file): slam/rgbd_stream.cc  <- slam_kit/slam/rgbd_offline.cc  (streaming ORB-SLAM3 runner)
                    slam/build_orbslam3.sh <- slam_kit/setup_orbslam3.sh
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_DIR = os.path.join(ROOT, "slam_kit_3", "slam_kit")


def _load(name):
    if KIT_DIR not in sys.path:
        sys.path.insert(0, KIT_DIR)
    return importlib.import_module(name)


def detector_module():
    return _load("detector")


def postprocess_module():
    return _load("postprocess")


def record_module():
    return _load("record")


def kit_model_path(rel="slam_kit_3/slam_kit/models/yolov8n.onnx"):
    return os.path.join(ROOT, rel)
