from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Global variables
camera = None
model = None
tracker = None
current_stats = {
    'available_slots': 0,
    'tracked_motors': 0,
    'last_update': datetime.now().strftime('%H:%M:%S')
}

# Simple Multi-Object Tracker Class
class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update(self, detections):
        if len(detections) == 0:
            self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
            for track in self.tracks:
                track['age'] += 1
            return []
        
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        if len(self.tracks) > 0:
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for d, detection in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = self.calculate_iou(detection[:4], track['box'])
            
            matches = []
            for d in range(len(detections)):
                for t in range(len(self.tracks)):
                    if iou_matrix[d, t] > self.iou_threshold:
                        matches.append((d, t, iou_matrix[d, t]))
            
            matches.sort(key=lambda x: x[2], reverse=True)
            
            matched_det_ids = set()
            matched_track_ids = set()
            
            for det_id, track_id, iou in matches:
                if det_id not in matched_det_ids and track_id not in matched_track_ids:
                    self.tracks[track_id]['box'] = detections[det_id][:4]
                    self.tracks[track_id]['confidence'] = detections[det_id][4]
                    self.tracks[track_id]['age'] = 0
                    self.tracks[track_id]['hits'] += 1
                    
                    matched_det_ids.add(det_id)
                    matched_track_ids.add(track_id)
            
            unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_ids]
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_track_ids]
        
        for det_id in unmatched_detections:
            new_track = {
                'id': self.track_id_count,
                'box': detections[det_id][:4],
                'confidence': detections[det_id][4],
                'age': 0,
                'hits': 1
            }
            self.tracks.append(new_track)
            self.track_id_count += 1
        
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
        
        self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
        
        result = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                x1, y1, x2, y2 = track['box']
                result.append([x1, y1, x2, y2, track['id']])
        
        return np.array(result)

# Configuration
PARKING_GRID = (100, 185, 810, 450)
SCALING_FACTOR = 1.0  # pixels per cm
SLOT_LENGTH_CM = 140  # Length of one parking slot in cm

def pixels_to_cm(pixels):
    return pixels / SCALING_FACTOR

def cm_to_pixels(cm):
    return cm * SCALING_FACTOR

def calculate_distance(center1, center2):
    distance_pixels = np.linalg.norm(np.array(center1) - np.array(center2))
    return pixels_to_cm(distance_pixels)

def detect_motor(frame):
    results = model(frame, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == 3 and confidence > 0.5:  # motorcycle class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2, y2, confidence])
    
    return np.array(detections) if detections else np.empty((0, 5))

def is_fully_in_parking_grid(box, grid):
    x_min, y_min, x_max, y_max = box
    px_min, py_min, px_max, py_max = grid
    return px_min <= x_min and px_max >= x_max and py_min <= y_min and py_max >= y_max

def draw_parking_grid(frame, grid):
    cv2.rectangle(frame, (grid[0], grid[1]), (grid[2], grid[3]), (0, 255, 0), 2)
    cv2.putText(frame, 'Parking Area', (grid[0], grid[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def calculate_available_slots(centers, parking_grid):
    if len(centers) == 0:
        parking_width_pixels = parking_grid[2] - parking_grid[0]
        parking_width_cm = pixels_to_cm(parking_width_pixels)
        total_slots = math.floor(parking_width_cm / SLOT_LENGTH_CM)
        return total_slots
    
    total_slots = 0
    left_boundary = parking_grid[0]
    right_boundary = parking_grid[2]
    
    sorted_centers = sorted(centers, key=lambda x: x[0])
    
    if len(sorted_centers) > 0:
        distance_pixels = abs(sorted_centers[0][0] - left_boundary)
        distance_cm = pixels_to_cm(distance_pixels)
        slots_before_first = math.floor(distance_cm / SLOT_LENGTH_CM)
        total_slots += slots_before_first
    
    for i in range(len(sorted_centers) - 1):
        distance_pixels = abs(sorted_centers[i+1][0] - sorted_centers[i][0])
        distance_cm = pixels_to_cm(distance_pixels)
        slots_between = math.floor(distance_cm / SLOT_LENGTH_CM)
        total_slots += slots_between
    
    if len(sorted_centers) > 0:
        distance_pixels = abs(right_boundary - sorted_centers[-1][0])
        distance_cm = pixels_to_cm(distance_pixels)
        slots_after_last = math.floor(distance_cm / SLOT_LENGTH_CM)
        total_slots += slots_after_last
    
    return total_slots

def draw_slot_visualization(frame, centers, parking_grid):
    left_boundary = parking_grid[0]
    right_boundary = parking_grid[2]
    parking_y_center = (parking_grid[1] + parking_grid[3]) // 2
    
    if len(centers) == 0:
        cv2.line(frame, (left_boundary, parking_y_center), 
                (right_boundary, parking_y_center), (0, 255, 0), 3)
        
        parking_width_pixels = right_boundary - left_boundary
        parking_width_cm = pixels_to_cm(parking_width_pixels)
        total_slots = math.floor(parking_width_cm / SLOT_LENGTH_CM)
        mid_point = ((left_boundary + right_boundary) // 2, parking_y_center - 20)
        cv2.putText(frame, f'Total Available: {total_slots} slots', 
                   mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return
    
    sorted_centers = sorted(centers, key=lambda x: x[0])
    
    if len(sorted_centers) > 0:
        first_motor_x = int(sorted_centers[0][0])
        distance_pixels = abs(first_motor_x - left_boundary)
        distance_cm = pixels_to_cm(distance_pixels)
        slots = math.floor(distance_cm / SLOT_LENGTH_CM)
        
        cv2.line(frame, (left_boundary, parking_y_center), 
                (first_motor_x, parking_y_center), (255, 0, 0), 2)
        
        mid_x = (left_boundary + first_motor_x) // 2
        cv2.putText(frame, f'{slots} slots', 
                   (mid_x - 30, parking_y_center - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    for i in range(len(sorted_centers) - 1):
        x1 = int(sorted_centers[i][0])
        x2 = int(sorted_centers[i+1][0])
        distance_pixels = abs(x2 - x1)
        distance_cm = pixels_to_cm(distance_pixels)
        slots = math.floor(distance_cm / SLOT_LENGTH_CM)
        
        cv2.line(frame, (x1, parking_y_center), (x2, parking_y_center), (255, 0, 0), 2)
        
        mid_x = (x1 + x2) // 2
        cv2.putText(frame, f'{slots} slots', 
                   (mid_x - 30, parking_y_center - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    if len(sorted_centers) > 0:
        last_motor_x = int(sorted_centers[-1][0])
        distance_pixels = abs(right_boundary - last_motor_x)
        distance_cm = pixels_to_cm(distance_pixels)
        slots = math.floor(distance_cm / SLOT_LENGTH_CM)
        
        cv2.line(frame, (last_motor_x, parking_y_center), 
                (right_boundary, parking_y_center), (255, 0, 0), 2)
        
        mid_x = (last_motor_x + right_boundary) // 2
        cv2.putText(frame, f'{slots} slots', 
                   (mid_x - 30, parking_y_center - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def initialize_camera():
    global camera
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera = cap
            print(f"Camera {i} opened successfully")
            return True
        cap.release()
    print("No camera found")
    return False

def initialize_model():
    global model, tracker
    try:
        model = YOLO("yolo11n.pt")
        tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        print("YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return False

def generate_frames():
    global current_stats
    frame_count = 0
    
    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            continue
            
        frame_count += 1
        
        # Draw parking grid
        frame = draw_parking_grid(frame, PARKING_GRID)
        
        # Detect motorcycles
        detections = detect_motor(frame)
        
        # Filter detections within parking grid
        filtered_detections = []
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if is_fully_in_parking_grid([x1, y1, x2, y2], PARKING_GRID):
                filtered_detections.append(detection)
        
        filtered_detections = np.array(filtered_detections) if filtered_detections else np.empty((0, 5))
        
        # Update tracker
        tracked_objects = tracker.update(filtered_detections)
        
        # Extract centers
        centers = []
        tracked_data = []
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            centers.append((x_center, y_center))
            tracked_data.append((x1, y1, x2, y2, track_id, x_center, y_center))
        
        # Draw tracked motorcycles
        for data in tracked_data:
            x1, y1, x2, y2, track_id, x_center, y_center = data
            
            color_idx = int(track_id) % 6
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[color_idx]
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'Motor ID:{int(track_id)}', 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (int(x_center), int(y_center)), 5, (255, 255, 0), -1)

        # Draw slot visualization
        draw_slot_visualization(frame, centers, PARKING_GRID)
        
        # Calculate available slots
        available_slots = calculate_available_slots(centers, PARKING_GRID)

        # Update global stats
        current_stats = {
            'available_slots': available_slots,
            'tracked_motors': len(tracked_objects),
            'last_update': datetime.now().strftime('%H:%M:%S')
        }

        # Add info overlay
        info_text = [
            f'Available Slots: {available_slots}',
            f'Tracked Motors: {len(tracked_objects)}',
            f'Frame: {frame_count}',
            f'Time: {current_stats["last_update"]}'
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return jsonify(current_stats)

if __name__ == '__main__':
    print("Initializing camera...")
    if not initialize_camera():
        print("Failed to initialize camera")
        exit(1)
    
    print("Initializing YOLO model...")
    if not initialize_model():
        print("Failed to initialize model")
        exit(1)
    
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)