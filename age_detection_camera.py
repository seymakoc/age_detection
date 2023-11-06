from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

# creating a method for detecting the faces and ages
def predict(frame, face_net, age_net, min_confidence=0.6):
    age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-24)', '(25,32)', '(38-43)', '(48-53)', '(60-100)']
    results = []
    
    # face detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # from face detections we're taking the faces with confidence greater than minimum confidence level we declared as 0.6
        if confidence > min_confidence:
            # getting the faces coordinates and save it into a variable and preprocessing it for the age prediction
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            face = frame[y1:y2, x1:x2]
            
            if face.shape[0] < 20 or face.shape[0] < 20:
                continue
            # making age predictions
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.43, 88.0, 115.0), swapRB=False)
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            
            i = age_preds[0].argmax()
            age = age_classes[i]
            age_confidence = age_preds[0][i]
            # making a dictionary for box coordinates and age detection results
            d = {"loc": (x1, y1, x2, y2), "age": (age, age_confidence)}
            results.append(d)
    # returning age detection results 
    return results


# loading the nets from pretrained models
print('Loading face detector model...')
face_detection_proto = './face_detector/opencv_face_detector.pbtxt'
face_detection_model = './face_detector/opencv_face_detector_uint8.pb'
face_net = cv2.dnn.readNet(face_detection_proto, face_detection_model)

print('Loading age detection model...')
age_detection_proto = './age_detector/age_deploy.prototxt'
age_detection_model = './age_detector/age_net.caffemodel'
age_net = cv2.dnn.readNet(age_detection_proto, age_detection_model)

# starting the camera
print('Starting the stream...')
stream = VideoStream(src=0).start()
time.sleep(2.0)

# we want to predict untill user press the 'q' key so a while loop for this
while True:
    # getting the frame from stream
    frame = stream.read()
    frame = imutils.resize(frame, width=400)
    # calling the predict method with frame and nets
    results = predict(frame, face_net, age_net)
    
    for r in results:
        # from results we get coordinates and predicted ages    
        x1, y1, x2, y2 = r['loc']
        image = frame[y1:y2, x1:x2]
        age_label, age_confidence = r['age']
        text = '{}: {:.2f}%'.format(age_label, age_confidence * 100)
        print(text)
        # showing the output to the user 
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# stopping the stream and everything
cv2.destroyAllWindows()
stream.stop()

