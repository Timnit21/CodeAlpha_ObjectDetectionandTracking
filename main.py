import cv2
from ultralytics import YOLO
import time

class ObjectDetectionTracker:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, output_path='output.avi'):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.output_path = output_path
        self.frame_count = 0
        
    def process_video(self, source=0, display=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30 # Default fps
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_start_time = time.time()
        total_objects_tracked = set()
        
        print("🚀 Starting detection. Press 'q' to stop.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = self.model.track(frame, persist=True, conf=self.conf_threshold, verbose=False)
                
                # Annotated Frame
                annotated_frame = results[0].plot()
                
                # Metrics
                self.frame_count += 1
                current_time = time.time()
                fps_val = 1 / (current_time - frame_start_time) if (current_time - frame_start_time) > 0 else 0
                frame_start_time = current_time
                
                # UI Overlays
                cv2.putText(annotated_frame, f'FPS: {int(fps_val)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(annotated_frame)
                if display:
                    cv2.imshow('Object Detection & Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"✅ Finished! Saved to {self.output_path}")

if __name__ == "__main__":
    tracker = ObjectDetectionTracker()
    tracker.process_video(source=0)