import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

confThreshold = 0.5
nmsThreshold = 0.3
whT = 320
image_directory = './Dataset'
classesFile = './coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = './yolov3.cfg'
modelWeights = './yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

bird_counts = []  # List to store the number of birds in each image

def findObjects(outputs, img):
    height, width = img.shape[:2]
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * width), int(det[3] * height)
                x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Count the number of birds in this image
    bird_count = len(indices)
    bird_counts.append(bird_count)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Lists to store data for plotting
image_numbers = list(range(1, len(os.listdir(image_directory)) + 1))

# Loop through images in the directory
for i, image_file in enumerate(sorted(os.listdir(image_directory))):
    image_path = os.path.join(image_directory, image_file)

    print(image_path)

    image = cv2.imread(image_path)

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False)
    net.setInput(blob)

    outputNames = net.getUnconnectedOutLayersNames()
    outputs = net.forward(outputNames)

    # Perform bird detection
    findObjects(outputs, image)
    print('i',i)

    # cv2.imshow('Image', image)
    # cv2.waitKey(7000)

# Plot the results
# plt.plot(image_numbers, bird_counts, marker='o')
# plt.xlabel('Image Number')
# plt.ylabel('Number of Birds')
# plt.title('Number of Birds in Each Image')
# plt.show()

# Close all OpenCV windows
# cv2.destroyAllWindows()
