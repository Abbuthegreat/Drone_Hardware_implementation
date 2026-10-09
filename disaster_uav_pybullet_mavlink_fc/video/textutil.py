"""Readable overlay text for the demo video. Outlines are drawn with the SAME font/thickness at pixel offsets: OpenCV lays glyphs out
differently for different thicknesses, so a thick black underlay drawn beneath thin coloured text does not line up (ghosting)."""
import cv2

_OFF = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]


def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1, outline=True):
    if outline:
        for dx, dy in _OFF:
            cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def dim_rect(img, x0, y0, x1, y1, alpha=0.55):
    """Darken a rectangle (semi-transparent backing for HUD text)."""
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (roi * (1.0 - alpha)).astype(img.dtype)
