import tkinter as tk
from tkinter import filedialog
import cv2

import cam_detect_video
import ee_detect_video
import textract

def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if video_path:
        save_video(video_path)

def save_video(video_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

root = tk.Tk()
root.title("Video Feeder")

select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

root.mainloop()

cam_detect_video.main()

ee_detect_video.main()

import cv2
import time

def process_video(video_path):
  """Processes a video frame by frame with a 1 second delay.

  Args:
    video_path: Path to the video file.
  """

  cap = cv2.VideoCapture(video_path)

  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    # Process the frame here (e.g., display it, apply filters)
    textract.preprocess_image(image_path=ret)
    cv2.waitKey(1)  # Adjust wait time as needed

    # Sleep for 1 second
    time.sleep(1)

  cap.release()
  cv2.destroyAllWindows()


process_video("video.mp4")
