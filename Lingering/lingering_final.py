import face_recognition
import cv2
import dlib
import numpy as np
import serial
import time
import sys

#Setup Communication path for arduino (In place of 'COM5' (windows) or ''/dev/tty.usbmodemxxx' (mac) put the port to which your arduino is connected)
arduino = serial.Serial(port='/dev/cu.usbmodem142101', baudrate=115200)

time.sleep(2)
print("Connected to Arduino...")

#importing the Haarcascade for face detectionqq
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
seungbo_image = face_recognition.load_image_file("seungbo.jpg")
#seungbo_image = face_recognition.load_image_file("Yongjoo.jpg")
seungbo_face_encoding = face_recognition.face_encodings(seungbo_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    seungbo_face_encoding,
#    biden_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
name = "Seungbo"
face_location = [0,0,0,0]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    matched = False

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face ㅂencodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        best_match_index = -1
        old_face_distance = 10
        for index in range(len(face_encodings)):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[index])
            #print("matches: ", matches)
            # # If a match was found in known_face_encodings, just use the first one.
            if matches[0]:
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[index])
                #print("face distance: ", face_distances, "index = ", index)
                if face_distances[0] < 0.4 and face_distances[0] < old_face_distance:
                    old_face_distance = face_distances[0]
                    best_match_index = index
                    matched = True

            #best_match_index = np.argmin(face_distances)
        #print("best_match_index = ", best_match_index)
 #           face_names.append(name)

    process_this_frame = not process_this_frame
 
    if matched:
#    if matches[best_match_index]:
#        name = known_face_names[best_match_index]
        top, right, bottom, left = face_locations[best_match_index]
 
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #Center of roi(Rectangle)
        xx = int(left+(right-left)/2)
        yy = int(top+(bottom - top)/2)
        # print (xx)

        center = (xx, yy)

        # sending data to arduino
        #print("Center of Rectangle is :", center)
        data = "X{0:d}Y{1:d}Z".format(xx, yy)
        #print ("output = '" +data+ "'")

        arduino.write(data.encode())

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
