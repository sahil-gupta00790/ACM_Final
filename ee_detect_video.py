from ultralytics import YOLO
import cv2
import csv
import os
from datetime import timedelta
import argparse

def door_detection(
    model_path,
    video_path,
    output_csv_path,
    confidence_threshold=0.2,
    frame_interval=1,
    save_frames=False,
    save_frames_path=None
):
    try:
        # Load the YOLO model
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
            csv_writer.writerow(['Timestamp', 'Frame', 'Door Detected', 'Confidence'])

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
                results = model(frame, conf=confidence_threshold)

                # Post-process the detection
                is_door = False
                max_confidence = 0

                if len(results[0].boxes) > 0:
                    is_door = True
                    for box in results[0].boxes:
                        confidence = box.conf.item()
                        if confidence > max_confidence:
                            max_confidence = confidence

                # Write results to CSV
                csv_writer.writerow([timestamp, f'entry_exit_frame_{frame_count:06d}', is_door, f'{max_confidence:.4f}'])

                # Optionally save the frame
                if save_frames and save_frames_path:
                    os.makedirs(save_frames_path, exist_ok=True)
                    frame_with_text = results[0].plot()
                    text = f"Door: {'Yes' if is_door else 'No'} ({max_confidence:.2f})"
                    cv2.putText(frame_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_door else (0, 0, 255), 2)
                    
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
    default_model_path = os.path.join("..","doors.pt")
    default_video_path = os.path.join("..","video.mp4")
    default_output_csv_path = os.path.join("..","output.csv")
    default_confidence = 0.2
    default_interval = 15
    default_save_frames = True
    default_frames_path = os.path.join(".." , "door_frames_output")

    #parser = argparse.ArgumentParser(description="Door Detection in Video")
    #parser.add_argument("--model-path", default=default_model_path, help="Path to the YOLO model file")
    #parser.add_argument("--video-path", default=default_video_path, help="Path to the input video file")
    #parser.add_argument("--output-csv-path", default=default_output_csv_path, help="Path to save the output CSV file")
    #parser.add_argument("--confidence", type=float, default=default_confidence, help="Confidence threshold for detection")
    #parser.add_argument("--interval", type=int, default=default_interval, help="Frame interval for processing")
    #parser.add_argument("--save-frames", action="store_true", default=default_save_frames, help="Save processed frames")
    #parser.add_argument("--frames-path", default=default_frames_path, help="Path to save processed frames")
#
    #args = parser.parse_args()

    door_detection(
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