import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import timedelta
import argparse

def cctv_detection(
    model_path,
    video_path,
    output_csv_path,
    confidence_threshold=0.2,
    frame_interval=10,
    save_frames=False,
    save_frames_path=None
):
    try:
        # Load the YOLOv5 model
        model = YOLO(model_path)

        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Prepare CSV file
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Frame', 'CCTV Detected', 'Confidence'])

            frame_count = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                # Calculate the correct timestamp
                seconds = (frame_count - 1) / fps
                timestamp = str(timedelta(seconds=seconds)).split('.')[0] + '.' + f"{seconds:.1f}".split('.')[1]

                # Perform detection
                results = model(frame)

                # Post-process the detection
                detections = results[0].boxes.data
                is_cctv = False
                max_confidence = 0

                for det in detections:
                    confidence = det[4].item()
                    if confidence > max_confidence:
                        max_confidence = confidence
                    if confidence > confidence_threshold:
                        is_cctv = True

                # Write results to CSV
                csv_writer.writerow([timestamp, f'cctv_frame_{frame_count:06d}', is_cctv, f'{max_confidence:.4f}'])

                # Optionally save the frame
                if save_frames and save_frames_path:
                    os.makedirs(save_frames_path, exist_ok=True)
                    frame_with_text = frame.copy()
                    text = f"CCTV: {'Yes' if is_cctv else 'No'} ({max_confidence:.2f})"
                    cv2.putText(frame_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_cctv else (0, 0, 255), 2)
                    
                    frame_filename = os.path.join(save_frames_path, f'frame_{frame_count:06d}.jpg')
                    cv2.imwrite(frame_filename, frame_with_text)

                print(f"Processed frame {frame_count}/{total_frames}")

        video.release()
        print(f"Detection completed. Results saved to {output_csv_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'video' in locals():
            video.release()

def main():
    # Default parameters
    default_model_path = os.path.join("..", "besttrain1.pt")
    default_video_path = os.path.join("..", "video.mp4")
    default_output_csv_path = os.path.join("..", "csv_output3.csv")

    default_confidence = 0.3
    default_interval = 15
    default_save_frames = True
    default_frames_path =os.path.join(".." , "vidsaves3")

    #parser = argparse.ArgumentParser(description="CCTV Detection in Video")
    #parser.add_argument("--model-path", default=default_model_path, help="Path to the YOLO model file")
    #parser.add_argument("--video-path", default=default_video_path, help="Path to the input video file")
    #parser.add_argument("--output-csv-path", default=default_output_csv_path, help="Path to save the output CSV file")
    #parser.add_argument("--confidence", type=float, default=default_confidence, help="Confidence threshold for detection")
    #parser.add_argument("--interval", type=int, default=default_interval, help="Frame interval for processing")
    #parser.add_argument("--save-frames", action="store_true", default=default_save_frames, help="Save processed frames")
    #parser.add_argument("--frames-path", default=default_frames_path, help="Path to save processed frames")

    #args = parser.parse_args()

    cctv_detection(
        model_path=default_model_path,
        video_path=default_video_path,
        output_csv_path=default_output_csv_path,
        confidence_threshold=default_confidence,
        frame_interval=default_interval,
        save_frames=default_save_frames,
        save_frames_path=default_frames_path
    )

if __name__ == "__main__":
    main()