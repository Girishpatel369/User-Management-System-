import cv2
import numpy as np

# Load YOLOv3 model and configuration files for vehicle detection
net = cv2.dnn.readNet("person.weights", "person.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the vehicle classes we want to detect
vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]

# Function to calculate the direction of movement
def calculate_direction(prev_box, current_box):
    # Extract centroid coordinates of previous and current bounding boxes
    prev_centerX = prev_box[0] + prev_box[2] / 2
    current_centerX = current_box[0] + current_box[2] / 2

    # Determine the direction based on the change in centroid position
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

    # Initialize lists to store detected vehicles and their bounding boxes
    vehicles = []
    boxes = []

    # Loop over each detection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                # Scale bounding box coordinates to image size
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Append the vehicle class and bounding box coordinates to the lists
                vehicles.append(classes[class_id])
                boxes.append((x, y, int(width), int(height)))

    return vehicles, boxes

# Function to draw bounding boxes around detected vehicles and display vehicle class and direction text
def draw_boxes(image, vehicles, boxes, directions):
    for (vehicle, (x, y, width, height), direction) in zip(vehicles, boxes, directions):
        color = (0, 255, 0) if direction == "Coming" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image, f"{vehicle} - {direction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to process live camera feed
def process_camera_feed():
    cap = cv2.VideoCapture("VID_20241203_125004368")
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
                # Find the nearest previous bounding box for each current bounding box
                distances = [np.linalg.norm(np.array(prev_box)[:2] - np.array(current_box)[:2]) for prev_box in prev_boxes]
                nearest_index = np.argmin(distances)
                prev_box = prev_boxes[nearest_index]
                
                # Calculate direction based on the movement of the vehicle
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
