import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

# Open a video capture object using the default camera (0)
video = cv2.VideoCapture(0)

# Load the Haar Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/haarcascade_frontalface_default.xml')

# Initialize an empty list to store face data
faces_data = []

# Counter to keep track of the number of frames processed
i = 0

# Get user input for their name
name = input("Enter your name and user_id : ")


# Loop to capture video frames and detect faces
while True:
    # Capture a frame from the video
    ret, frame = video.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]

        # Resize the cropped face image to 50x50 pixels
        resized_img = cv2.resize(crop_img, (50, 50))

        # Append the resized face image to the faces_data list every 5 frames
        if len(faces_data) <= 10 and i % 10 == 0:
            faces_data.append(resized_img)

        i = i + 1

        # Display the count of captured faces on the frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the current frame with annotations
    cv2.imshow("Frame", frame)

    # Wait for a key press or until 5 faces are captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 10:
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array and reshape it
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(10, -1)

# Check if 'names.pkl' is present in the 'Data/' directory
if 'names.pkl' not in os.listdir('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data'):
    # If not present, create a list with the entered name repeated 5 times
    names = [name] * 10
    # Save the list to 'names.pkl'
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # If 'names.pkl' is present, load the existing list
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    # Append the entered name 5 times to the existing list
    names = names + [name] * 10
    # Save the updated list to 'names.pkl'
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Check if 'faces_data.pkl' is present in the 'Data/' directory
if 'faces_data.pkl' not in os.listdir('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data'):
    # If not present, save the NumPy array 'faces_data' to 'faces_data.pkl'
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # If 'faces_data.pkl' is present, load the existing array
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    # Append the new array 'faces_data' to the existing array
    faces = np.append(faces, faces_data, axis=0)
    # Save the updated array to 'faces_data.pkl'
    with open('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)





# import cv2
# import pickle
# import numpy as np
# import os
# import time
# from datetime import datetime

# # Open a video capture object using the default camera (0)
# video = cv2.VideoCapture(0)

# # Load the Haar Cascade Classifier for face detection
# facedetect = cv2.CascadeClassifier('C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/haarcascade_frontalface_default.xml')

# # Initialize an empty list to store face data
# faces_data = []
# user_data = {}  # Dictionary to store user ID and name

# # Counter to keep track of the number of frames processed
# i = 0

# # Get user input for their name and ID
# name = input("Enter your name: ")
# user_id = input("Enter your ID: ")

# # Loop to capture video frames and detect faces
# while True:
#     # Capture a frame from the video
#     ret, frame = video.read()

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     # Iterate over detected faces
#     for (x, y, w, h) in faces:
#         # Crop the face region from the frame
#         crop_img = frame[y:y+h, x:x+w, :]

#         # Resize the cropped face image to 50x50 pixels
#         resized_img = cv2.resize(crop_img, (50, 50))

#         # Append the resized face image to the faces_data list every 5 frames
#         if len(faces_data) < 10 and i % 10 == 0:
#             faces_data.append(resized_img)
#             user_data[user_id] = name

#         i += 1

#         # Display the count of captured faces on the frame
#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

#     # Display the current frame with annotations
#     cv2.imshow("Frame", frame)

#     # Wait for a key press or until 5 faces are captured
#     k = cv2.waitKey(1)
#     if k == ord('q') or len(faces_data) == 10:
#         break

# # Release the video capture object and close all windows
# video.release()
# cv2.destroyAllWindows()

# # Convert the list of face images to a NumPy array and reshape it
# faces_data = np.asarray(faces_data)
# faces_data = faces_data.reshape(len(faces_data), -1)

# # Check if 'names.pkl' is present in the 'Data/' directory
# data_dir = 'C:/Users/shrut/OneDrive/Desktop/9 Smart-Attendence-System/9 Smart-Attendence-System/Data/'
# if 'names.pkl' not in os.listdir(data_dir):
#     # If not present, save the dictionary user_data to 'names.pkl'
#     with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
#         pickle.dump(user_data, f)
# else:
#     # If 'names.pkl' is present, load the existing dictionary
#     with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
#         existing_user_data = pickle.load(f)
#     # Update the dictionary with the new user ID and name
#     existing_user_data[user_id] = name
#     # Save the updated dictionary to 'names.pkl'
#     with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
#         pickle.dump(existing_user_data, f)

# # Check if 'faces_data.pkl' is present in the 'Data/' directory
# if 'faces_data.pkl' not in os.listdir(data_dir):
#     # If not present, save the NumPy array 'faces_data' to 'faces_data.pkl'
#     with open(os.path.join(data_dir, 'faces_data.pkl'), 'wb') as f:
#         pickle.dump(faces_data, f)
# else:
#     # If 'faces_data.pkl' is present, load the existing array
#     with open(os.path.join(data_dir, 'faces_data.pkl'), 'rb') as f:
#         existing_faces = pickle.load(f)
#     # Append the new array 'faces_data' to the existing array
#     updated_faces = np.append(existing_faces, faces_data, axis=0)
#     # Save the updated array to 'faces_data.pkl'
#     with open(os.path.join(data_dir, 'faces_data.pkl'), 'wb') as f:
#         pickle.dump(updated_faces, f)
