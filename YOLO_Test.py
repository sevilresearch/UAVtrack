import numpy as np
import airsim
import cv2
import yolov7

# Load the trained YOLO model
net = cv2.dnn.readNet("C:/Users/hjaye/PycharmProjects/AirSim Project/yolov7/yolov7.weights", "C:/Users/hjaye/PycharmProjects/AirSim Project/yolov7/yolov7.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to get the output layer names
def getOutputsNames(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()


while True:
    # Get an image from AirSim
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    # Process the image
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    # Prepare the frame to be fed to the network
    blob = cv2.dnn.blobFromImage(img_rgb, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                center_x = int(detection[0] * response.width)
                center_y = int(detection[1] * response.height)
                width = int(detection[2] * response.width)
                height = int(detection[3] * response.height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                cv2.rectangle(img_rgb, (left, top), (left + width, top + height), (255, 0, 0), 2)

    # Display the image with detections
    cv2.imshow("Image", img_rgb)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()