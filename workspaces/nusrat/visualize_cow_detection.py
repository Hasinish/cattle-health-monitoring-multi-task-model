import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
INPUT_VIDEO_PATH = r"D:\T25301094 P2\test_cow.mp4"
OUTPUT_VIDEO_PATH = r"D:\T25301094 P2\test_cow_detection.mp4"
MODEL_NAME = "yolov8n.pt"  # Lightweight YOLOv8 Nano model
MAX_FRAMES_TO_PROCESS = 2000  # Default limit to process a subset of the long video quickly

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize YOLO Cow Detection on Video")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save annotated video")
    parser.add_argument("--limit", type=int, default=MAX_FRAMES_TO_PROCESS, help="Maximum number of frames to process (set to 0 for full video)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found at: {args.input}")

    print(f"Loading YOLOv8 detector: {MODEL_NAME}...")
    # Load pre-trained YOLOv8-Nano model
    detector = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {args.input}")

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set frame limit
    limit = args.limit if args.limit > 0 else total_frames
    limit = min(limit, total_frames)

    print(f"\nProcessing Video:")
    print(f"  Source:       {args.input}")
    print(f"  Resolution:   {width}x{height}")
    print(f"  FPS:          {fps}")
    print(f"  Total Frames: {total_frames} (Limit set to process first {limit} frames)")
    print(f"  Saving to:    {args.output}\n")

    # Define VideoWriter to save output with annotations
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    show_window = True

    # Process frames
    for i in tqdm(range(limit), desc="Drawing bounding boxes"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = detector(frame, verbose=False)[0]

        # Draw box for any detected cow
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Class ID 19 is 'cow' in COCO dataset
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw green bounding box around cow
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw text label with confidence score
                label = f"Cow {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the annotated frame to output video
        out.write(frame)

        # Display the annotated frame in real-time if GUI window support is available
        if show_window:
            try:
                cv2.imshow("Real-Time YOLO Cow Detection (Press 'q' to Quit)", frame)
                # Check for 'q' key press to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nVisualization stopped early by user.")
                    break
            except cv2.error:
                print("\nWarning: Headless environment or OpenCV without GUI support detected.")
                print("Running in HEADLESS mode. Video is still being written to the output file.")
                show_window = False

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print(f"\nFinished! Annotated video saved successfully to: {args.output}")

if __name__ == "__main__":
    main()
