import os
import cv2


# image = cv2.imread('faces.jpg')
from django.conf import settings
base_dir = settings.BASE_DIR

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

class_names = []

class_file = os.path.join(base_dir, 'scripts\object_detection', 'coco-labels')
print(class_file)
with open(class_file, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

config_path = os.path.join(base_dir, 'scripts\object_detection','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weights_path = os.path.join(base_dir, 'scripts\object_detection', 'frozen_inference_graph.pb')

net = cv2.dnn_DetectionModel(weights_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()

    class_ids, confs, bbox = net.detect(image, confThreshold=0.5)
    # print(class_ids, bbox)
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, class_names[class_id-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow('Object Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

