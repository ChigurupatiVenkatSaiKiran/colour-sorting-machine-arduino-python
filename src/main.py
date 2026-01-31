import cv2
from color_detector import ColorDetector


def start_colour_sorting() -> None:
    detector = ColorDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            detected_colour = detector.detect_colour(frame)
            label = detected_colour if detected_colour else "None"

            cv2.putText(
                frame,
                f"Detected Colour: {label}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Colour Sorting Machine", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_colour_sorting()