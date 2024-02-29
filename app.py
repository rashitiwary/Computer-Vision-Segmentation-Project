from flask import Flask, render_template
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import io


app = Flask(__name__)


def findObjects(outputs, img, bird_counts):
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




def get_images():
    im_names = []
    bird_counts = []  # List to store the number of birds in each image
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
        findObjects(outputs, image,bird_counts)
        
        cv2.imwrite(os.path.join('static', f"{i}.png"), image)
        im_names.append(f'{i}.png')
        
        
    return im_names, bird_counts, image_numbers

def get_graphs(bird_counts, image_numbers):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(image_numbers, bird_counts)
    axis.set_xlabel('Image Number')
    axis.set_ylabel('Number of Birds')
    axis.set_title('Number of Birds in Each Image')
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    
    with open(os.path.join('static', 'graph.png'), 'wb') as f:
        f.write(output.getvalue())


@app.route('/')
def gallery():
    images, bird_counts, image_numbers = get_images()
    get_graphs(bird_counts, image_numbers)
    image_collage = ['Bird_beach_1.jpeg','Bird_beach_2.jpeg','Bird_beach_3.jpeg','Bird_beach_4.jpeg','Bird_beach_5.jpeg']
    return render_template("index.html",images=images, image_collage=image_collage)
        
if __name__ == '__main__':
    app.run(debug=False)
        