// rgbd_offline.cc - run ORB-SLAM3 (RGB-D) on a folder recorded by record.py
//
// Usage: rgbd_offline <ORBvoc.txt> <settings.yaml> <run_dir> [viewer 0|1] [speed]
//   speed = playback speed factor (1.0 = real time, 0.5 = half speed, 0 = as fast as possible)
//           Real time (or slower) is recommended: it gives local mapping time to keep up.
//
// Output (in run_dir):
//   CameraTrajectory.txt    per-frame camera poses Twc, largest map only
//   KeyFrameTrajectory.txt  keyframe poses Twc
//   Format per line: timestamp_ns tx ty tz qx qy qz qw   (EuRoC style, timestamp * 1e9)

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 4) {
        cerr << "Usage: rgbd_offline <ORBvoc.txt> <settings.yaml> <run_dir> [viewer 0|1] [speed]" << endl;
        return 1;
    }
    const string runDir = argv[3];
    const bool useViewer = argc > 4 ? atoi(argv[4]) != 0 : false;
    const double speed = argc > 5 ? atof(argv[5]) : 1.0;

    // associations.txt: t_rgb rgb/xxx.jpg t_depth depth/xxx.png
    ifstream fa(runDir + "/associations.txt");
    if (!fa.is_open()) { cerr << "Cannot open " << runDir << "/associations.txt" << endl; return 1; }
    vector<double> vTs;
    vector<string> vRgb, vDepth;
    string line;
    while (getline(fa, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        double t1, t2; string a, b;
        ss >> t1 >> a >> t2 >> b;
        if (a.empty() || b.empty()) continue;
        vTs.push_back(t1); vRgb.push_back(a); vDepth.push_back(b);
    }
    const int N = (int)vTs.size();
    if (N == 0) { cerr << "No frames found." << endl; return 1; }
    cout << "Frames: " << N << "  viewer=" << useViewer << "  speed=" << speed << endl;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, useViewer);
    const float imageScale = SLAM.GetImageScale();

    int nOk = 0, nLost = 0;
    double tSum = 0;
    for (int i = 0; i < N; i++) {
        cv::Mat im = cv::imread(runDir + "/" + vRgb[i], cv::IMREAD_UNCHANGED);
        cv::Mat depth = cv::imread(runDir + "/" + vDepth[i], cv::IMREAD_UNCHANGED);
        if (im.empty() || depth.empty()) { cerr << "Skip unreadable frame " << i << endl; continue; }
        if (imageScale != 1.f) {
            int w = im.cols * imageScale, h = im.rows * imageScale;
            cv::resize(im, im, cv::Size(w, h));
            cv::resize(depth, depth, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
        }

        auto t0 = chrono::steady_clock::now();
        SLAM.TrackRGBD(im, depth, vTs[i]);
        auto t1 = chrono::steady_clock::now();
        double ttrack = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();
        tSum += ttrack;

        int st = SLAM.GetTrackingState();   // 2 = OK
        if (st == 2) nOk++; else nLost++;
        if (i % 100 == 0 || i == N - 1)
            printf("[%5d/%d] state=%d  ok=%d  not_ok=%d  avg_track=%.1f ms\n", i, N, st, nOk, nLost, 1000.0 * tSum / (i + 1));

        if (speed > 0 && i < N - 1) {
            double dt = (vTs[i + 1] - vTs[i]) / speed - ttrack;
            if (dt > 0) this_thread::sleep_for(chrono::microseconds((long)(dt * 1e6)));
        }
    }

    // Let local mapping / loop closing finish a bit
    this_thread::sleep_for(chrono::seconds(2));
    SLAM.Shutdown();

    SLAM.SaveTrajectoryEuRoC(runDir + "/CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC(runDir + "/KeyFrameTrajectory.txt");
    printf("Done. Tracking OK in %d / %d frames (%.1f%%)\n", nOk, N, 100.0 * nOk / N);
    return 0;
}
