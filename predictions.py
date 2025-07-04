

import cv2
from ultralytics import YOLO
from agastya import sendtoTelegram

# Initialize YOLO with the Model Name
model = YOLO("best.pt")

# Define the action for low-confidence cases
def take_action(label, confidence, frame):
    print(f"Action Required: {label} detected with low confidence ({confidence:.2f}%).")
    sendtoTelegram(f"Low confidence detected for {label}: {confidence:.2f}%")
    
    # Save the current frame as an image
    fpath="./demo.jpg"
    
    print("Captured and saved the frame as demo.jpeg")

# Class names mapping from the model
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
            # confidence = box.conf * 100  # Convert to percentage
            label = class_names[label_id]  # Get class name
            
            # Check for "protective helmet" with confidence < 20%

            print(f"--------Confi = {confidence} , {type(confidence)}and Label = {label}")
            if label == "Protective Helmet" and confidence < 45:
                cv2.imwrite("demo.jpg", frame)
                take_action(label, confidence, frame)


            print(f"label : {label} and confi : {confidence}")
    
    # Display the video feed
    cv2.imshow("Webcam", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
