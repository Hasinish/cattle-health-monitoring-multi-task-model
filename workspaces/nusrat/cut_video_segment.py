import os
import cv2
from tqdm import tqdm

# Configuration
INPUT_VIDEO = r"D:\T25301094 P2\full_download.mp4"
OUTPUT_VIDEO = r"D:\T25301094 P2\cut_cow_video_2.mp4"
START_SEC = 175.0  # 2:55 (2 * 60 + 55)
END_SEC = 215.0    # 3:35 (3 * 60 + 35)

def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Input video not found at: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Trimming segment: {START_SEC}s (2:55) to {END_SEC}s (3:35)")
    print(f"Frames: {start_frame} to {end_frame} (Total to cut: {end_frame - start_frame})")

    for _ in tqdm(range(start_frame, end_frame), desc="Trimming video"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"\nSuccessfully saved trimmed segment to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
