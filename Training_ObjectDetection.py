import airsim
import cv2
import numpy as np

# Initialize AirSim Client
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# load YOLO v7
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load the classes from coco.naems file
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

def get_image_from_airsim():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype = np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return img_rgb

def detect_objects(image):
    height = width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen (class id, confidence, bounding box coordinates)
    class_ids = []
    confidence = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object Detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle Coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    