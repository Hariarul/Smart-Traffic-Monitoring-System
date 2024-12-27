import streamlit as st
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import time
import os
from PIL import Image
import tempfile

# Load YOLO model and Keras models
model = YOLO("yolov8n-seg.pt")
color_model = load_model("car_color_red_blue.h5")
gender_model = load_model("gender_male_female.h5")

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Streamlit UI components
st.title("Real-time Object Detection with YOLOv8")
st.markdown("This app detects cars, people, and other vehicles.")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Process image
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(uploaded_file)
        img = np.array(image)
        img_resized = cv2.resize(img, (1280, 720))

        results = model(img_resized, stream=True)
        car_count, male_count, female_count, other_vehicle_count = 0, 0, 0, 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf
                if conf < 0.3: continue

                currentClass = classNames[cls]
                w, h = x2 - x1, y2 - y1

                # Car detection and color classification
                if currentClass == 'car':
                    car_roi = img_resized[y1:y2, x1:x2]
                    car_roi_resized = cv2.resize(car_roi, (150, 150)) / 255.0
                    car_color = np.argmax(color_model.predict(np.expand_dims(car_roi_resized, axis=0)))
                    car_color_label = "Blue" if car_color == 0 else "Red"
                    car_count += 1
                    cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(0, 255, 0))
                    cvzone.putTextRect(img_resized, f"Car: {car_color_label}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

                # Person detection and gender classification
                elif currentClass == 'person':
                    person_roi = img_resized[y1:y2, x1:x2]
                    person_roi_resized = cv2.resize(person_roi, (150, 150)) / 255.0
                    gender = np.argmax(gender_model.predict(np.expand_dims(person_roi_resized, axis=0)))
                    gender_label = "Male" if gender == 0 else "Female"
                    male_count += (gender == 0)
                    female_count += (gender == 1)
                    cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(255, 0, 0))
                    cvzone.putTextRect(img_resized, f"{gender_label}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

                # Other vehicle detection
                elif currentClass in ["motorbike", "bus", "train", "truck", "bicycle", "boat"]:
                    other_vehicle_count += 1
                    cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(0, 0, 255))
                    cvzone.putTextRect(img_resized, f"{currentClass.capitalize()}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

        # Display the image
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB", use_column_width=True)

    # Process video
    elif uploaded_file.type in ["video/mp4", "video/avi", "video/mov"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while True:
            success, img = cap.read()
            if not success: break

            img_resized = cv2.resize(img, (1280, 720))
            results = model(img_resized, stream=True)

            car_count, male_count, female_count, other_vehicle_count = 0, 0, 0, 0

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    conf = box.conf
                    if conf < 0.3: continue

                    currentClass = classNames[cls]
                    w, h = x2 - x1, y2 - y1

                    # Car detection and color classification
                    if currentClass == 'car':
                        car_roi = img_resized[y1:y2, x1:x2]
                        car_roi_resized = cv2.resize(car_roi, (150, 150)) / 255.0
                        car_color = np.argmax(color_model.predict(np.expand_dims(car_roi_resized, axis=0)))
                        car_color_label = "Blue" if car_color == 0 else "Red"
                        car_count += 1
                        cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(0, 255, 0))
                        cvzone.putTextRect(img_resized, f"Car: {car_color_label}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

                    # Person detection and gender classification
                    elif currentClass == 'person':
                        person_roi = img_resized[y1:y2, x1:x2]
                        person_roi_resized = cv2.resize(person_roi, (150, 150)) / 255.0
                        gender = np.argmax(gender_model.predict(np.expand_dims(person_roi_resized, axis=0)))
                        gender_label = "Male" if gender == 0 else "Female"
                        male_count += (gender == 0)
                        female_count += (gender == 1)
                        cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(255, 0, 0))
                        cvzone.putTextRect(img_resized, f"{gender_label}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

                    # Other vehicle detection
                    elif currentClass in ["motorbike", "bus", "train", "truck", "bicycle", "boat"]:
                        other_vehicle_count += 1
                        cvzone.cornerRect(img_resized, (x1, y1, w, h), l=5, t=3, colorR=(0, 0, 255))
                        cvzone.putTextRect(img_resized, f"{currentClass.capitalize()}", (x1, y1 - 10), scale=1, thickness=2, offset=5)

            # Display counts
            cv2.putText(img_resized, f"Cars: {car_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_resized, f"Males: {male_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img_resized, f"Females: {female_count}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img_resized, f"Other Vehicles: {other_vehicle_count}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert and display the processed frame
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            stframe.image(img_rgb, channels="RGB", use_column_width=True)

        cap.release()
