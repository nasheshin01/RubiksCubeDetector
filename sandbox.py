from detector import YoloNasDetector
from PIL import Image
import cv2

def show_detected_objects():
    detector = YoloNasDetector()

    # OpenCV setup
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Detection', 1280, 720)
    cap = cv2.VideoCapture(0)  # Change the index if using a different camera

    while True:
        ret, frame = cap.read()

        if ret:
            # Convert OpenCV frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform object detection
            detections = detector.detect(pil_image)
            labels = ['cube', 'cube_in_hands']
            colors = [(0, 255, 0), (255, 0, 0)]

            # Draw bounding boxes on the frame
            for (x1, y1, x2, y2), confidence, label_idx in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[label_idx], 2)
                cv2.putText(frame, f'{labels[label_idx]} ({round(confidence, 2)})', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[label_idx], 2)

            # Display the frameq
            cv2.imshow('Object Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Run the script
show_detected_objects()