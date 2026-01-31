import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ColorDetector:
    """
    Detects the dominant colour in an image frame using HSV thresholds.
    """

    def __init__(self, pixel_threshold: int = 1500) -> None:
        self.pixel_threshold = pixel_threshold
        self._colour_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "Red": (
                np.array([0, 120, 70]),
                np.array([10, 255, 255]),
            ),
            "Green": (
                np.array([36, 50, 70]),
                np.array([89, 255, 255]),
            ),
            "Blue": (
                np.array([90, 50, 70]),
                np.array([128, 255, 255]),
            ),
        }

    def detect_colour(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect dominant colour in the given frame.

        :param frame: Input image in BGR format
        :return: Colour name or None
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for colour, (lower, upper) in self._colour_ranges.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            if cv2.countNonZero(mask) > self.pixel_threshold:
                return colour

        return None