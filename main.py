from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import os
import numpy as np
import cv2
from math import pow, sqrt

def detect_and_predict_mask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	#facenet detect face
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)


	faces = []
	locs = []
	preds = []

	# loop detections
	for i in range(0, detections.shape[2]):
        #config confidence sensitive to detect
		confidence = detections[0, 0, i, 2]
		if confidence > 0.4:

		
            #config to find w,h,x,y
            
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# check the frame is within the frame dimension
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract roi of face and convert from bgt to rgb
			# resize to 224px * 224px and calculate
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add a face and frame the bounding box
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only if at least one face is detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32) #simulate how to wear a mask and calculate the predict value

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
prototxtPath = "deploy.prototxt"
weightsPath = "model/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# model to simulate the appearance of wearing a mess
maskNet = load_model("model/maskmodel.h5") #load model

#start
print("Starting Video Stream...")


labels = [line.strip() for line in open("class_labels.txt")]

bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))


# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe("model/SSD_MobileNet_prototxt.txt", "model/SSD_MobileNet.caffemodel")

#clip video or using a webcam
cap = cv2.VideoCapture('videos/test1.mp4')
frame_no = 0

while 1: #loop

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800) #resize to 800px
    (h, w) = frame.shape[:2]

    #resize the frame to suit the model 300px*300px
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    network.setInput(blob)
    detections = network.forward()

    pos_dict = dict()
    coordinates = dict()
    for (box, pred) in zip(locs, preds):
        # bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask,improper) = pred
        
        #if wearing a mask
        if (mask > withoutMask and mask > improper):
            label = "Mask"
            color = (0, 255, 0)
        #if wearing a mask but only wearing it to the chin or not wearing it to the nose
        elif (withoutMask > mask and withoutMask > improper):
            label = "improper"
            color = (255, 0, 0)
        #ืnot wearing
        elif (improper > mask and improper > withoutMask):
            label = "No Mask"
            color = (0, 0, 255)

        label = "{}: {:.2f}% ".format(label, max(mask, withoutMask, improper) * 100)

        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    # focal length
    F = 615

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:

            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # filtering only persons detected in the frame. Class Id of 'person' is 15
            if class_id == 15.00:

                # drawing bounding box for object
                cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)
                label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                print("{}".format(label))


                coordinates[i] = (startX, startY, endX, endY)

                # mid point bounding box
                x_mid = round((startX+endX)/2,4)
                y_mid = round((startY+endY)/2,4)

                height = round(endY-startY,4)

                # triangle distance from camera
                distance = (165 * F)/height
                print("Distance(cm):{dist}\n".format(dist=distance))

                # center point of bounding boxes (cm.)
                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

    # distance between every detected object in the frame.
    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # check distance less than 2 meters or 200 centimeters
                if dist < 200:
                    close_objects.add(i)
                    close_objects.add(j)

    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = (0,0,255) # improper
        else:
            COLOR = (0,255,0) # mask
        (startX, startY, endX, endY) = coordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        #convert cm > feet
        cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)



    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # show frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # press q to exit
    if key == ord("q"):
        break

# clean
cap.release()
cv2.destroyAllWindows()
