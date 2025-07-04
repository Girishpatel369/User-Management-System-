import tkinter as tk
from PIL import Image, ImageTk
import subprocess  # For running external scripts
import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
import pyttsx3
import threading
import sendmail
import random

engine = pyttsx3.init()

import cv2
from ultralytics import YOLO
from agastya import sendtoTelegram,sendtoTelegram1

# Initialize YOLO with the Model Name
model = YOLO("best.pt")

# Define the action for low-confidence cases
def take_action(label, confidence, frame,msg):
    print(f"Action Required: {label} detected with low confidence ({confidence:.2f}%).")
    sendtoTelegram(f"{msg} Low confidence detected for {label}: {confidence:.2f}%")
    
    # Save the current frame as an image
    fpath="./demo.jpg"
    
    print("Captured and saved the frame as demo.jpeg")
    

def take_action_oneway(msg):
    
    
    # Save the current frame as an image
    sendtoTelegram1("One way voilation detected")
    fpath="./oneway-violation.jpg"
    
    print("Captured and saved the frame one way voilation.jpg")

# Class names mapping from the model

def voicebuzzer():
    engine.say('crime detected')
    engine.runAndWait()
    
def run_helmet_detection():
    # Assuming you have already loaded your model
    class_names = model.names  # Assumes the model has a `names` attribute

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform prediction on the current frame
        results = model.predict(source=frame, show=False, conf=0.35)

        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                confidence_tensor = box.conf  # Confidence as a tensor
                confidence = confidence_tensor.item() * 100  # Convert to percentage

                label_id = int(box.cls)  # Class ID
                label = class_names[label_id]  # Get class name
                
                # Only process if the label is "Protective Helmet"
                if label == "Protective Helmet":
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers

                    # Draw bounding box for "Protective Helmet" with green color
                    color = (0, 255, 0)  # Green for helmet
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw the rectangle
                    cv2.putText(frame, f"{label}: {confidence:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    print(f"Confidence = {confidence}, Label = {label}")

                    # If confidence is below 45%, take action
                    if confidence < 45:
                        cv2.imwrite("demo.jpg", frame)
                        take_action(label, confidence, frame, "No helmet detected")

        # Display the video feed with bounding boxes for helmets
        cv2.imshow("Webcam", frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def run_one_way_detection():
    # Load YOLOv3 model and configuration files for vehicle detection
    net = cv2.dnn.readNet("person.weights", "person.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Define the vehicle classes we want to detect
    vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

    # Function to calculate the direction of movement
    def calculate_direction(prev_box, current_box):
        prev_centerX = prev_box[0] + prev_box[2] / 2
        current_centerX = current_box[0] + current_box[2] / 2

        if current_centerX > prev_centerX:
            direction = "Coming"
        else:
            direction = "Going"
        
        return direction

    # Function to detect vehicles using YOLOv3
    def detect_vehicles(image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

        vehicles = []
        boxes = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in vehicle_classes:
                    box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    vehicles.append(classes[class_id])
                    boxes.append((x, y, int(width), int(height)))

        return vehicles, boxes

    # Function to draw bounding boxes around detected vehicles and save cropped images for "Going" direction
    def draw_boxes(image, vehicles, boxes, directions):
        for (vehicle, (x, y, width, height), direction) in zip(vehicles, boxes, directions):
            color = (0, 255, 0) if direction == "Coming" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)

            if direction == "Going":
                caption = "Rules violated"
                cv2.putText(image, f"{vehicle} - {caption}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Crop the part of the image containing the "Going" vehicle
                crop_img = image[y:y + height, x:x + width]

                # Save only the cropped image of the "Going" vehicle
                n = random.randint(1, 100)
                fpath = f"./violation/case{n}.jpg"
                cv2.imwrite(fpath, crop_img)

            elif direction == "Coming":
                pass

    # Function to process live camera feed
    def process_camera_feed():
        cap = cv2.VideoCapture("newinput.mp4")
        ret, prev_frame = cap.read()
        prev_vehicles, prev_boxes = [], []
        while cap.isOpened():
            ret, current_frame = cap.read()
            if not ret:
                break

            current_vehicles, current_boxes = detect_vehicles(current_frame)
            if current_vehicles and prev_vehicles:
                directions = []
                for current_box in current_boxes:
                    distances = [np.linalg.norm(np.array(prev_box)[:2] - np.array(current_box)[:2]) for prev_box in prev_boxes]
                    nearest_index = np.argmin(distances)
                    prev_box = prev_boxes[nearest_index]

                    direction = calculate_direction(prev_box, current_box)
                    directions.append(direction)

                draw_boxes(current_frame, current_vehicles, current_boxes, directions)
                print(current_vehicles, directions)

            cv2.imshow("Detected Vehicles", current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_vehicles, prev_boxes = current_vehicles, current_boxes

        cap.release()
        cv2.destroyAllWindows()

    # Process live camera feed
    process_camera_feed()


def run_weapon_detection():
    
    #subprocess.Popen(['python', 'weapondetection.py'])
    net = cv2.dnn.readNet("savedmodel/yolo_training_2000.weights", "savedmodel/yolo.cfg")
    classes = ["Weapon"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    count=0
    engine.say("Weapon detection activated")
    engine.runAndWait()
    
    cap=cv2.VideoCapture(0)
    
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        # width = 512
        # height = 512

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #print(indexes)
        if indexes == 0: print("weapon detected in frame")
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                
                n=random.randint(1,100)
                fpath=f"./weapon/case{n}.jpg"
                cv2.imwrite(fpath, img)
                # threading.Thread(target=voicebuzzer).start()
                # count+=1
                print(count)
                
                #if count==1:
                # subject = "Crime Detected"
                # sendmail.sendalert("weapon.jpg",subject)
                #     #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
        #counter=0

        # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Create the main application window
root = tk.Tk()
root.title("Smart City Traffic Observation")

# Load and display background image
bg_image = Image.open("background.png")
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create header label
header_label = tk.Label(root, text="Smart City Traffic Observation", font=("Helvetica", 24), bg='white')
header_label.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

# Create buttons for running specific scripts
helmet_button = tk.Button(root, text="Helmet Detection", font=("Helvetica", 16), command=run_helmet_detection)
helmet_button.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

oneway_button = tk.Button(root, text="One Way Detection", font=("Helvetica", 16), command=run_one_way_detection)
oneway_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

weapon_button = tk.Button(root, text="Weapon Detection", font=("Helvetica", 16), command=run_weapon_detection)
weapon_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

# Start the main event loop
root.mainloop()
