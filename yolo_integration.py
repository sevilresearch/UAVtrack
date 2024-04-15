import airsim
import cv2
import numpy as np
import quaternion

# Initialize AirSim client
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

# Load YOLO v7
net = cv2.dnn.readNet("C:/Users/hjaye/PycharmProjects/AirSim Project/yolov7/yolov7.weights", "C:/Users/hjaye/PycharmProjects/AirSim Project/yolov7/yolov7.cfg")
layer_names = net.getLayerNames()
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except Exception:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def get_image_from_airsim():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return img_rgb


def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen (class id, confidence, bounding box coordinates)
    class_ids = []
    confidences = []
    boxes = []
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

        # Use Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            if label == 'cylinder':  # assuming 'cylinder' is in coco.names
                color = colors[class_ids[i]]
                cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
                              color, 2)
                cv2.putText(image, label, (boxes[i][0], boxes[i][1] - 10), font, 1, color, 2)

    return image


while True:
    img = get_image_from_airsim()
    detected_img = detect_objects(img)
    cv2.imshow('Detected Image', detected_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()