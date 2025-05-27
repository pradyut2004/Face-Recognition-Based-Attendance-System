# attendance_system.py

import cv2
import numpy as np
import os
from datetime import datetime
import dlib

def start_recognition():
    # --- Configuration ---
    KNOWN_FACES_DIR = "known_faces"
    ATTENDANCE_FILE = "attendance.csv"

    # --- Load Dlib Models ---
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # --- Storage for known face encodings and names ---
    known_faces = []
    known_names = []

    # --- Load and encode known face images ---
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = detector(rgb_image)

                if detections:
                    shape = shape_predictor(rgb_image, detections[0])
                    face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
                    known_faces.append(np.array(face_descriptor))
                    known_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"⚠️ No face found in {filename}, skipping.")
            except Exception as e:
                print(f"[ERROR] Could not process {filename}: {e}")

    # --- Marked attendance tracker ---
    marked_names = set()

    def mark_attendance(name):
        if name not in marked_names:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            with open(ATTENDANCE_FILE, 'a') as f:
                f.write(f"{name},{time_string}\n")
            marked_names.add(name)

    # --- Start Webcam ---
    video_capture = cv2.VideoCapture(0)
    print("Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        detections = detector(rgb_small_frame)
        for det in detections:
            shape = shape_predictor(rgb_small_frame, det)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)
            face_encoding = np.array(face_descriptor)

            # Compare with known faces
            distances = np.linalg.norm(known_faces - face_encoding, axis=1)
            min_distance_index = np.argmin(distances)

            if distances[min_distance_index] < 0.6:  # Threshold for match
                name = known_names[min_distance_index]
                mark_attendance(name)

                # Scale back up the face location
                top, right, bottom, left = det.top(), det.right(), det.bottom(), det.left()
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    video_capture.release()
    cv2.destroyAllWindows()

# Optional: if you want to test this file directly (without GUI)
if __name__ == "__main__":
    start_recognition()
