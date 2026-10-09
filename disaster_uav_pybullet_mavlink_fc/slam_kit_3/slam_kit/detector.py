"""
detector.py - YOLOv8 person detector with three interchangeable backends:

  1. ultralytics  (GPU via PyTorch/TensorRT if installed; accepts .pt / .engine / .onnx)
  2. onnxruntime  (pip install onnxruntime  -> CPU; onnxruntime-gpu -> CUDA)
  3. OpenCV DNN   (no extra install - uses the OpenCV that ships with JetPack)

backend="auto" tries them in that order.  All return a list of (x1, y1, x2, y2, conf).
"""
import os
import numpy as np
import cv2

PERSON = 0


def _letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, px, py


def _decode(out, r, px, py, conf, iou, w, h):
    """YOLOv8 raw output (1, 84, N) -> person boxes in original image coordinates."""
    out = np.squeeze(out)
    if out.shape[0] < out.shape[1]:
        out = out.T                        # (N, 84)
    scores = out[:, 4 + PERSON]
    m = scores > conf
    if not np.any(m):
        return []
    b, s = out[m, :4], scores[m]
    x1 = (b[:, 0] - b[:, 2] / 2 - px) / r
    y1 = (b[:, 1] - b[:, 3] / 2 - py) / r
    bw, bh = b[:, 2] / r, b[:, 3] / r
    rects = [[float(a), float(c), float(d), float(e)] for a, c, d, e in zip(x1, y1, bw, bh)]
    keep = cv2.dnn.NMSBoxes(rects, s.astype(float).tolist(), conf, iou)
    keep = np.array(keep).flatten() if len(keep) else []
    dets = []
    for k in keep:
        x, y, ww, hh = rects[k]
        dets.append((max(0.0, x), max(0.0, y), min(w - 1.0, x + ww), min(h - 1.0, y + hh), float(s[k])))
    return dets


class Detector:
    def __init__(self, model="models/yolov8n.onnx", conf=0.4, iou=0.5, imgsz=640, backend="auto"):
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.backend = None
        errors = []
        order = ["ultralytics", "onnxruntime", "opencv"] if backend == "auto" else [backend]
        for b in order:
            try:
                getattr(self, f"_init_{b}")(model)
                self.backend = b
                break
            except Exception as e:
                errors.append(f"{b}: {e}")
        if self.backend is None:
            raise RuntimeError("No detector backend worked:\n  " + "\n  ".join(errors))
        print(f"[detector] backend = {self.backend}  model = {model}")

    # --- backends ---
    def _init_ultralytics(self, model):
        from ultralytics import YOLO
        import torch
        # Without CUDA ultralytics is no faster than onnxruntime - prefer the lighter ones then
        if not torch.cuda.is_available() and not model.endswith(".engine"):
            raise RuntimeError("torch has no CUDA")
        pt = model
        if model.endswith(".onnx") and os.path.exists(model[:-5] + ".pt"):
            pt = model[:-5] + ".pt"
        self.m = YOLO(pt)
        self.device = 0

    def _init_onnxruntime(self, model):
        import onnxruntime as ort
        prov = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()]
        prov = [p for p in prov if p != "TensorrtExecutionProvider"]  # TRT build takes ages first run
        self.sess = ort.InferenceSession(model, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        print(f"[detector] onnxruntime providers: {self.sess.get_providers()}")

    def _init_opencv(self, model):
        self.net = cv2.dnn.readNetFromONNX(model)
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            pass
        # sanity run (fails fast if this OpenCV can't run the model)
        self._run_opencv(np.zeros((480, 640, 3), np.uint8))

    # --- inference ---
    def _run_opencv(self, img):
        lb, r, px, py = _letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        return _decode(self.net.forward(), r, px, py, self.conf, self.iou, img.shape[1], img.shape[0])

    def __call__(self, img):
        h, w = img.shape[:2]
        if self.backend == "ultralytics":
            res = self.m.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=[PERSON],
                                 device=self.device, verbose=False, half=True)[0]
            return [(float(a), float(b), float(c), float(d), float(s))
                    for (a, b, c, d), s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        if self.backend == "onnxruntime":
            lb, r, px, py = _letterbox(img, self.imgsz)
            x = (lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
            out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
            return _decode(out, r, px, py, self.conf, self.iou, w, h)
        return self._run_opencv(img)
