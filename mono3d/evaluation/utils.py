import cv2
import numpy as np

def draw_3D_box(img, corners, color = (255, 255, 0)):
    """
        draw 3D box in image with OpenCV,
        the order of the corners should be the same with BBox3dProjector
    """
    points = np.array(corners[0:2], dtype=np.int32) #[2, 8]
    points = [tuple(points[:,i]) for i in range(8)]
    for i in range(1, 5):
        cv2.line(img, points[i], points[(i % 4 + 1)], color, 2)
        cv2.line(img, points[(i + 4) % 8], points[((i) % 4 + 5) % 8], color, 2)
    cv2.line(img, points[2], points[7], color)
    cv2.line(img, points[3], points[6], color)
    cv2.line(img, points[4], points[5],color)
    cv2.line(img, points[0], points[1], color)
    return img