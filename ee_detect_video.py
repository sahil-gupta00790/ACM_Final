import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
import os
from datetime import timedelta
import csv

from ultralytics import YOLO

class VideoAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analysis Tool")
        self.root.geometry("1000x800")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.create_widgets()
        
        self.video_path = None
        self.processing = False
        
        # Load YOLO models
        self.cctv_model = YOLO(os.path.join("..", "besttrain1.pt"))
        self.door_model = YOLO(os.path.join("..", "doors.pt"))
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video selection
        select_button = ttk.Button(main_frame, text="Select Video", command=self.select_video)
        select_button.pack(pady=10)
        
        # Video display
        self.video_canvas = tk.Canvas(main_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(pady=10)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Analysis", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            self.results_text.insert(tk.END, f"Selected video: {self.video_path}\n")
            self.start_button.config(state=tk.NORMAL)
    
    def start_analysis(self):
        if not self.video_path:
            return
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def stop_analysis(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        cctv_csv_path = os.path.join("..", "cctv_output.csv")
        door_csv_path = os.path.join("..", "door_output.csv")
        
        with open(cctv_csv_path, 'w', newline='') as cctv_csv, open(door_csv_path, 'w', newline='') as door_csv:
            cctv_writer = csv.writer(cctv_csv)
            door_writer = csv.writer(door_csv)
            
            cctv_writer.writerow(['Timestamp', 'Frame', 'CCTV Detected', 'Confidence'])
            door_writer.writerow(['Timestamp', 'Frame', 'Door Detected', 'Confidence'])
            
            while cap.isOpened() and self.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                seconds = (frame_count - 1) / fps
                timestamp = str(timedelta(seconds=seconds)).split('.')[0] + '.' + f"{seconds:.1f}".split('.')[1]
                
                # CCTV detection
                cctv_results = self.cctv_model(frame)
                cctv_detections = cctv_results[0].boxes.data
                is_cctv = False
                cctv_confidence = 0
                
                for det in cctv_detections:
                    confidence = det[4].item()
                    if confidence > cctv_confidence:
                        cctv_confidence = confidence
                    if confidence > 0.2:
                        is_cctv = True
                
                cctv_writer.writerow([timestamp, f'cctv_frame_{frame_count:06d}', is_cctv, f'{cctv_confidence:.4f}'])
                
                # Door detection
                door_results = self.door_model(frame, conf=0.2)
                is_door = len(door_results[0].boxes) > 0
                door_confidence = max([box.conf.item() for box in door_results[0].boxes]) if is_door else 0
                
                door_writer.writerow([timestamp, f'door_frame_{frame_count:06d}', is_door, f'{door_confidence:.4f}'])
                
                # Update results display
                self.update_results(timestamp, is_cctv, cctv_confidence, is_door, door_confidence)
                
                # Display frame
                self.display_frame(frame, is_cctv, cctv_confidence, is_door, door_confidence)
                
                print(f"Processed frame {frame_count}/{total_frames}")
                
                time.sleep(0.1)  # Adjust delay as needed
        
        cap.release()
        self.results_text.insert(tk.END, "Analysis completed.\n")
        self.stop_analysis()
    
    def update_results(self, timestamp, is_cctv, cctv_confidence, is_door, door_confidence):
        self.results_text.insert(tk.END, f"Timestamp: {timestamp}\n")
        self.results_text.insert(tk.END, f"CCTV detected: {'Yes' if is_cctv else 'No'} (Confidence: {cctv_confidence:.2f})\n")
        self.results_text.insert(tk.END, f"Door detected: {'Yes' if is_door else 'No'} (Confidence: {door_confidence:.2f})\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        self.results_text.see(tk.END)
    
    def display_frame(self, frame, is_cctv, cctv_confidence, is_door, door_confidence):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        
        cctv_text = f"CCTV: {'Yes' if is_cctv else 'No'} ({cctv_confidence:.2f})"
        door_text = f"Door: {'Yes' if is_door else 'No'} ({door_confidence:.2f})"
        
        cv2.putText(frame, cctv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_cctv else (0, 0, 255), 2)
        cv2.putText(frame, door_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_door else (0, 0, 255), 2)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.video_canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalysisGUI(root)
    root.mainloop()