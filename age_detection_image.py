import numpy as np
import cv2
import argparse
import os
import glob as glob

# creating a folder for the output images
os.makedirs(os.path.join('C:/projects/age_detection', 'output'), exist_ok=True)

# creating the argument for the input images
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--image', required=True, help='path to input image folder')
args = vars(arg_parser.parse_args())

# declaring the classes for the age spans
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25,32)', '(38-43)', '(48-53)', '(60-100)']

# loading face and age detector nets with pretrained models
print('Loading face detector model...')
face_detection_proto = './face_detector/opencv_face_detector.pbtxt'
face_detection_model = './face_detector/opencv_face_detector_uint8.pb'
face_net = cv2.dnn.readNet(face_detection_proto, face_detection_model)

print('Loading age detection model...')
age_detection_proto = './age_detector/age_deploy.prototxt'
age_detection_model = './age_detector/age_net.caffemodel'
age_net = cv2.dnn.readNet(age_detection_proto, age_detection_model)

# from the provided argument for inputs getting the test images
DIR_TEST = args['image']
test_images = []
if os.path.isdir(DIR_TEST):
    image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
    for file_type in image_file_types:
        test_images.extend(glob.glob(f'{DIR_TEST}/{file_type}'))
else:
    test_images.append(DIR_TEST)
print(f'Test instances: {len(test_images)}')

for i in range(len(test_images)):
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    # reading the image and preprocessing it for the face detection model
    image = cv2.imread(test_images[i])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # detecting the face
    print('Detecting the face...')
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # from face detections we're taking the faces with confidence greater than 0.7
        if confidence > 0.7:
            # getting the faces coordinates and save it into a variable and preprocessing it for the age prediction
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            face = image[y1:y2, x1:x2]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # predicting the age 
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            age = age_classes[i]
            age_confidence = age_preds[0][i]
            text = '{}: {:.2f}%'.format(age, age_confidence * 100)

            # showing outputs and saving them the output directory
            print('{}'.format(text))
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow('Image', image)
            cv2.waitKey(1)
            cv2.imwrite(f'C:/projects/age_detection/output/{image_name}.jpg', image)

# closing the cv2 windows
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
