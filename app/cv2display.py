# cv2display.py
import cv2
import numpy as np


class CV2Display:
    def __init__(self, window_name: str = "RGB Video"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def draw_overlay(
        self, rgb_frame: np.ndarray, depth_frame: np.ndarray
    ) -> np.ndarray:
        """
        Draw depth information and a central cross marker on the RGB image
        """
        rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        center_y, center_x = depth_frame.shape[0] // 2, depth_frame.shape[1] // 2
        depth_center = depth_frame[center_y, center_x]
        text = f"Depth at Centre: {depth_center}"
        cv2.putText(
            rgb_bgr, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3
        )
        cv2.drawMarker(
            rgb_bgr,
            (rgb_bgr.shape[1] // 2, rgb_bgr.shape[0] // 2),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=4,
        )
        return rgb_bgr

    def show(self, frame: any) -> None:
        cv2.imshow(self.window_name, frame)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
