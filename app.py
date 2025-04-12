# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLOv3 model and configurations
MODEL_PATH = 'yolov3.weights'
CONFIG_PATH = 'yolov3.cfg'
LABELS_PATH = 'coco.names'

# Categories mapping with expanded electronics items
CATEGORIES = {
    "Mobile Devices": [
        "mobile phone", "cell phone", "smartphone", "iphone", 
        "android phone", "mobile device"
    ],
    
    "Computing Devices": [
        "laptop", "mouse", "keyboard", "monitor", 
        "computer", "desktop", "tablet", "touchpad",
        "pc", "notebook", "macbook", "chromebook"
    ],
    
    "Audio Devices": [
        "headphones", "earbuds", "earphones", 
        "wireless earbuds", "speaker", "bluetooth speaker",
        "headset", "airpods", "earpods", "microphone",
        "audio device", "sound system"
    ],
    
    "Power & Charging": [
        "power adapter", "charger", "power supply",
        "power bank", "battery pack", "usb charger",
        "ac adapter", "power cord", "charging cable",
        "wireless charger", "charging pad", "battery charger"
    ],
    
    "Accessories": [
        "usb drive", "flash drive", "cable",
        "hdmi cable", "usb cable", "ethernet cable",
        "adapter", "dongle", "hub", "card reader",
        "memory card", "storage device", "external drive"
    ],
    
    "Home Electronics": [
        "tv", "television", "remote", "remote control",
        "refrigerator", "fridge", "air conditioner",
        "microwave", "router", "wifi router", "smart tv",
        "gaming console", "thermostat", "smart display"
    ],
    
    "Wearable Tech": [
        "smartwatch", "fitness tracker", 
        "smart glasses", "wireless headset",
        "smart band", "apple watch", "smart ring",
        "wearable device"
    ],
    
    "Photography & Video": [
        "camera", "digital camera", "webcam",
        "video camera", "action camera", "security camera",
        "camera lens", "tripod", "flash", "camera accessory"
    ]
}

def load_model():
    """Load YOLO model and class labels"""
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Load class labels
    with open(LABELS_PATH, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
        
    return net, CLASSES

def detect_objects(image, net, CLASSES):
    """Detect objects in the image using YOLO"""
    # Get image dimensions
    (H, W) = image.shape[:2]
    
    # Create blob and set input
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass
    detections = net.forward(output_layers)
    
    results = []
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Lower confidence threshold for better detection of smaller items
            if confidence > 0.4:  # Adjusted threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression with adjusted thresholds
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            label = CLASSES[class_ids[i]].lower()
            
            # Find category for the detected object
            category = "Unknown"
            for cat, items in CATEGORIES.items():
                # Check if the label matches any item in the category
                if any(item in label or label in item for item in items):
                    category = cat
                    break
                
                # Additional check for similar terms
                for item in items:
                    if (len(item) > 3 and (
                        item in label or 
                        label in item or 
                        any(word in label.split() for word in item.split())
                    )):
                        category = cat
                        break
            
            # Add to results if it's an electronic item
            if category != "Unknown":
                # Clean up the label
                display_label = label.title()  # Capitalize the label
                # Remove common prefixes if present
                prefixes = ["a ", "an ", "the "]
                for prefix in prefixes:
                    if display_label.lower().startswith(prefix):
                        display_label = display_label[len(prefix):]
                
                results.append({
                    "label": display_label,
                    "confidence": confidences[i],
                    "category": category,
                    "box": boxes[i]
                })
    
    return results

# Load model at startup
net, CLASSES = load_model()

@app.route('/detect', methods=['POST'])
def detect():
    """Handle detection requests"""
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Convert base64 to image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image and get detections
        results = detect_objects(image, net, CLASSES)
        
        return jsonify({
            "success": True,
            "detections": results
        })
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")  # Log the error
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Check if model files exist
    required_files = [MODEL_PATH, CONFIG_PATH, LABELS_PATH]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found!")
            exit(1)
    
    print("Starting Electronics Detector server...")
    app.run(debug=True, port=5000)