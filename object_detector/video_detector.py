from flask import Blueprint, render_template, request, send_file, current_app
from ultralytics import YOLO
import cv2
import os
import pandas as pd
from collections import Counter

video_detector_blueprint = Blueprint('video_detector', __name__, template_folder='templates')

# Initialize YOLO model
model = YOLO("yolov8n.pt")

@video_detector_blueprint.route('/video-detector', methods=['GET', 'POST'])
def video_detector():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save uploaded video
            filename = secure_filename(file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Process video
            cap = cv2.VideoCapture(upload_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            output_filename = f"detected_{filename}"
            output_path = os.path.join(current_app.config['UPLOAD_FOLDER'], output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            object_counter = Counter()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                # Count objects
                classes = results[0].boxes.cls
                for cls in classes:
                    object_name = model.names[int(cls)]
                    object_counter[object_name] += 1

                frame_count += 1

            cap.release()
            out.release()

            # Save object counts
            csv_filename = f"counts_{filename.split('.')[0]}.csv"
            csv_path = os.path.join(current_app.config['UPLOAD_FOLDER'], csv_filename)
            pd.DataFrame.from_dict(object_counter, orient='index').to_csv(csv_path)

            return render_template('video_detector.html',
                                original_video=upload_path,
                                processed_video=output_path,
                                csv_file=csv_path,
                                frame_count=frame_count,
                                object_counts=object_counter)
    
    return render_template('video_detector.html')

@video_detector_blueprint.route('/download-video/<filename>')
def download_video(filename):
    return send_file(
        os.path.join(current_app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True
    )

@video_detector_blueprint.route('/download-csv/<filename>')
def download_csv(filename):
    return send_file(
        os.path.join(current_app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True,
        mimetype='text/csv'
    )