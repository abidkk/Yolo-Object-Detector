# ====================== [IMPORTS] ======================
import os
import time
import cv2
import pandas as pd
from PIL import Image
from collections import Counter
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, session, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
from flask import send_from_directory

# ====================== [FLASK CONFIGURATION] ======================
app = Flask(__name__)
app.secret_key = 'abc123'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model once
model = YOLO("yolov8n.pt")
# model = YOLO("yolov5s.pt")

# ====================== [ROUTES] ======================

@app.route('/')
def home():
    return render_template('home.html')


# ====================== [IMAGE DETECTION] ======================
@app.route('/image-detector', methods=['GET', 'POST'])
def image_detector():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        results = model(upload_path)
        result_filename = f"detected_{filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        results[0].save(filename=result_path)

        detected_objects = []
        boxes = results[0].boxes

        if boxes is not None:
            class_ids = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for i in range(len(class_ids)):
                class_id = int(class_ids[i])
                object_name = model.names.get(class_id, "Unknown")
                detected_objects.append({
                    'object': object_name,
                    'confidence': float(confidences[i]),
                    'x_min': float(xyxy[i][0]),
                    'y_min': float(xyxy[i][1]),
                    'x_max': float(xyxy[i][2]),
                    'y_max': float(xyxy[i][3])
                })

        df = pd.DataFrame(detected_objects)
        csv_filename = f"detections_{filename.split('.')[0]}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        df.to_csv(csv_path, index=False)

        session['original_image'] = upload_path
        session['processed_image'] = result_path
        session['csv_path'] = csv_path
        session['detection_count'] = len(detected_objects)

        return render_template('image_detector.html',
                               original_image=upload_path,
                               processed_image=result_path,
                               detection_count=len(detected_objects),
                               detected_objects=detected_objects)

    return render_template('image_detector.html')


# ====================== [VIDEO DETECTION] ======================
@app.route('/video-detector', methods=['GET', 'POST'])
def video_detector():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        cap = cv2.VideoCapture(upload_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        slow_fps = max(1.0, original_fps / 2)

        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, slow_fps, (width, height))

        object_counter = Counter()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            if boxes is not None:
                class_ids = boxes.cls.cpu().numpy()
                for cls_id in class_ids:
                    object_name = model.names.get(int(cls_id), "Unknown")
                    object_counter[object_name] += 1

            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()
        time.sleep(1)

        object_data_df = pd.DataFrame([object_counter]).T.reset_index()
        object_data_df.columns = ["Object", "Count"]

        csv_filename = f"object_counts_{filename.split('.')[0]}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        object_data_df.to_csv(csv_path, index=False)

        session['original_video'] = upload_path
        session['processed_video'] = output_path
        session['csv_path'] = csv_path
        session['object_data'] = object_data_df.to_dict('records')
        session['frame_count'] = frame_count

        return render_template('video_detector.html',
                               original_video=upload_path,
                               processed_video=output_path,
                               frame_count=frame_count,
                               object_data=object_data_df.to_dict('records'))

    return render_template('video_detector.html')


# ====================== [REAL-TIME WEBCAM DETECTION] ======================
video_capture = None
recording_writer = None
# after processing the video
# output_path = 'processed/output.mp4'
# Save the path in session
# session['processed_video'] = output_path 

def generate_frames():
    global video_capture, recording_writer

    video_capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_output.mp4')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_output.mp4')
    print(f"Saving video to: {output_path}")

    fps = 20.0
    frame_size = (640, 480)
    recording_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        recording_writer.write(annotated_frame)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
    recording_writer.release()
    session['processed_video'] = os.path.join(app.config['UPLOAD_FOLDER'], output_path)
    
    
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/start-webcam')
def start_webcam():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop-webcam')
def stop_webcam():
    global video_capture, recording_writer
    if video_capture:
        video_capture.release()
    if recording_writer:
        recording_writer.release()
    return "Webcam stopped"


# ====================== [DOWNLOAD ROUTES] ======================

@app.route('/download-image')
def download_image():
    if 'processed_image' in session:
        return send_file(session['processed_image'], as_attachment=True)
    return redirect(url_for('image_detector'))

@app.route('/download-image-csv')
def download_image_csv():
    if 'csv_path' in session:
        return send_file(
            session['csv_path'],
            as_attachment=True,
            mimetype='text/csv',
            download_name=f"image_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    return redirect(url_for('image_detector'))

@app.route('/download-video')
def download_video():
    if 'processed_video' in session:
        return send_file(session['processed_video'], as_attachment=True)
    return redirect(url_for('video_detector'))



@app.route('/download-video-csv')
def download_video_csv():
    if 'csv_path' in session:
        return send_file(
            session['csv_path'],
            as_attachment=True,
            mimetype='text/csv',
            download_name=f"video_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    return redirect(url_for('video_detector'))

@app.route('/download-csv')
def download_csv():
    if 'csv_path' in session:
        return send_file(session['csv_path'], as_attachment=True)
    return redirect(url_for('video_detector'))


# @app.route('/download-camvideo')
# def download_camvideo():
#     if 'camvideo_processed_path' in session:
#         return send_file(session['camvideo_processed_path'], as_attachment=True)
#     return redirect(url_for('webcam'))
    
# @app.route('/download-camvideo')
# def download_camvideo():
#     if 'processed_video' in session:
#         return send_file(session['processed_video'], as_attachment=True)
#     return redirect(url_for('webcam'))

@app.route('/download-camvideo')
def download_camvideo():
    # Define the directory and filename
    # directory = 'static/uploads'
    directory = os.path.join(app.config['UPLOAD_FOLDER'])
    filename = "webcam_output.mp4"
    

    
    # Send the file as an attachment (forces download)
    return send_from_directory(
        directory=directory,
        path=filename,
        as_attachment=True
    )



# @app.route('/download-video')
# def download_video():
#     if 'processed_video' in session:
#         return send_file(session['processed_video'], as_attachment=True)
#     return redirect(url_for('webcam.html'))


# ====================== [APP RUN] ======================
if __name__ == '__main__':
    app.run(debug=True)
