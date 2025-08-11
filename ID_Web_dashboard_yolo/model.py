from ultralytics import YOLO
import cv2
import pandas as pd
import threading
import time
import sys

# Shared control flags and data
is_running = False               # Flag to start/stop tracking
csv_data = []                    # List to hold detection data

# Dictionary to keep track of already seen object IDs by class
seen_ids = {
    0: set(),  # person_with_id
    1: set(),  # person_without_id
    2: set(),  # bike
    3: set()   # car
}

# Mapping YOLO class IDs to human-readable labels
CLASS_NAMES = {
    0: "person_with_id",
    1: "person_without_id",
    2: "bike",
    3: "car"
}

# Load the trained YOLO model (update with your actual model path if different)
# model = YOLO("custom_yolo_v3.pt")
model = YOLO("custom_yolov11l.pt")  ## change model here

# Function to save detection data to a CSV file
def save_csv():
    df = pd.DataFrame(csv_data)
    df.to_csv("object_counts.csv", index=False)

    # Print a simple summary in console
    print(f"\n📊 Final Count Summary:")
    for class_id, label in CLASS_NAMES.items():
        print(f"🔹 {label}: {len(seen_ids[class_id])}")
    print(f"📦 Total Unique Objects: {sum(len(ids) for ids in seen_ids.values())}")

# Function that continuously reads frames and applies tracking
def tracking_loop():
    global is_running, csv_data, seen_ids

    # cap = cv2.VideoCapture(0)  # Start webcam
    # print("📹 Webcam started...")

 
    video_path = "idvideo2.mp4"  # Replace with your actual path
    cap = cv2.VideoCapture(video_path)
    print("🎥 Video file opened...")


    while is_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO object tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if results.boxes.id is None:
            continue

        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        xyxy = results.boxes.xyxy.cpu().numpy()

        for i in range(len(ids)):
            class_id = classes[i]
            track_id = ids[i]
            label = CLASS_NAMES.get(class_id, "unknown")

            # Avoid logging the same object ID multiple times
            if track_id not in seen_ids[class_id]:
                seen_ids[class_id].add(track_id)
                csv_data.append({
                    "track_id": track_id,
                    "label": label,
                    "timestamp_sec": round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
                })

            # Draw bounding box and label on frame
            x1, y1, x2, y2 = map(int, xyxy[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}-{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Show the frame
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Tracking stopped.")
    save_csv()

# Function to start tracking in a new thread
def start_tracking():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=tracking_loop, daemon=True).start()

# Function to stop tracking
def stop_tracking():
    global is_running
    is_running = False
