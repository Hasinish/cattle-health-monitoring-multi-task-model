import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cut_cow_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cropped_cow_video.mp4")
MODEL_NAME = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")  # Lightweight YOLOv8 Nano model
TARGET_SIZE = (224, 224)   # Standard crop size for neural networks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crop Cow Detections from Video and Filter Backgrounds")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save cropped video")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of frames to process (0 for full video)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found at: {args.input}")

    print(f"Loading YOLOv8 detector: {MODEL_NAME}...")
    detector = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {args.input}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    limit = args.limit if args.limit > 0 else total_frames
    limit = min(limit, total_frames)

    print(f"\nProcessing Video:")
    print(f"  Source:       {args.input}")
    print(f"  FPS:          {fps}")
    print(f"  Total Frames: {total_frames} (Processing limit: {limit})")
    print(f"  Output size:  {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  Saving to:    {args.output}\n")
    print("All slides, texts, and frames without cows will be automatically filtered out!\n")

    # Define VideoWriter to save output (size is fixed to 224x224)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, TARGET_SIZE)

    show_window = True
    cropped_count = 0

    for i in tqdm(range(limit), desc="Cropping cows"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = detector(frame, verbose=False)[0]

        cow_crop = None
        best_box = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Class ID 19 is 'cow' in COCO dataset
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cow_crop = frame[y1:y2, x1:x2]

        # If a cow was found, resize, save, and display it
        if cow_crop is not None and cow_crop.size > 0:
            resized_crop = cv2.resize(cow_crop, TARGET_SIZE)
            out.write(resized_crop)
            cropped_count += 1

            if show_window:
                try:
                    cv2.imshow("Cropped Cow Stream (Press 'q' to Quit)", resized_crop)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped early by user.")
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
    
    print(f"\nFinished!")
    print(f"Saved {cropped_count} cropped frames to: {args.output}")

if __name__ == "__main__":
    main()
