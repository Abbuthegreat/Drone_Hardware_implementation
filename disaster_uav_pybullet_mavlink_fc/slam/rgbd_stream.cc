// rgbd_stream.cc - STREAMING variant of the attached kit's slam_kit/slam/rgbd_offline.cc (ORB-SLAM3 RGB-D runner).
//
// The kit runner replays a recorded folder (associations.txt + rgb/*.jpg + depth/*.png). The simulation needs an ONLINE runner, so the
// frame loop is kept as-is (same System construction, same TrackRGBD call, same tracking-state convention: 2 = OK) but frames arrive
// over stdin and one pose line per frame goes to stdout.
//
// Built by slam/build_orbslam3.sh (Ubuntu 22.04). stdout of the process is reserved for the protocol below; ORB-SLAM3 chatter goes to stderr.
//
// Usage:  rgbd_stream <ORBvoc.txt> <settings.yaml> [viewer 0|1]
//
// stdin  (binary, little endian), repeated:
//     char[4] "FRM1" | double t_seconds | int32 width | int32 height |
//     uint8 bgr[width*height*3] | uint16 depth_mm[width*height]       (0 = invalid)
// stdout (text), one line per frame:
//     POSE <t> <state> <tx> <ty> <tz> <qx> <qy> <qz> <qw> <map_id>   Twc (camera -> world of Atlas map <map_id>), state 2 = tracking OK
// A frame with t < 0 ends the session.

#include <cstdio>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

static bool readExact(void *dst, size_t n)
{
    return fread(dst, 1, n, stdin) == n;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        cerr << "Usage: rgbd_stream <ORBvoc.txt> <settings.yaml> [viewer 0|1]" << endl;
        return 1;
    }
    const bool useViewer = argc > 3 ? atoi(argv[3]) != 0 : false;

    // ORB-SLAM3 prints banners / status text to stdout. Keep a private copy of the real stdout for the pose protocol
    // and point fd 1 at stderr so nothing else can corrupt it.
    FILE *proto = fdopen(dup(1), "w");
    dup2(2, 1);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, useViewer);   // same construction as rgbd_offline.cc
    fprintf(proto, "READY\n");
    fflush(proto);

    while (true) {
        char magic[4];
        double t;
        int32_t w, h;
        if (!readExact(magic, 4) || memcmp(magic, "FRM1", 4) != 0) break;
        if (!readExact(&t, sizeof(t)) || !readExact(&w, 4) || !readExact(&h, 4)) break;
        if (t < 0) break;
        cv::Mat im(h, w, CV_8UC3), depth(h, w, CV_16UC1);
        if (!readExact(im.data, (size_t)w * h * 3) || !readExact(depth.data, (size_t)w * h * 2)) break;

        Sophus::SE3f Tcw = SLAM.TrackRGBD(im, depth, t);      // same call as rgbd_offline.cc
        int st = SLAM.GetTrackingState();                       // 2 = OK (kit convention)
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f p = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        // id of the Atlas map the pose belongs to (public API: map of a tracked map point). After a tracking loss ORB-SLAM3 may start a NEW map with
        // its own origin; the consumer must not treat poses of different maps as one frame. -1 = unknown (no tracked point).
        int map_id = -1;
        if (st == 2) {
            std::vector<ORB_SLAM3::MapPoint*> mps = SLAM.GetTrackedMapPoints();
            for (ORB_SLAM3::MapPoint *mp : mps) {
                if (mp && !mp->isBad() && mp->GetMap()) { map_id = (int)mp->GetMap()->GetId(); break; }
            }
        }
        fprintf(proto, "POSE %.6f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d\n", t, st, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w(), map_id);
        fflush(proto);
    }
    SLAM.Shutdown();
    return 0;
}
