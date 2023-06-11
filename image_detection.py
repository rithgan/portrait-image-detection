import os
import sys
import cv2
import numpy as np

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

__location__ = os.getcwd()

folder_path = __location__ + '/images'
file_paths = []

def check_portrait_image(image_path):
    print("Checking following image", image_path)
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except NameError:
        print("Image is not in correct format")
        return
    except cv2.error:
        print("Image is not in correct format")
        return

    # Calculate the mean value of the grayscale image
    mean_value = np.mean(gray)
    print(mean_value)
    # Define a threshold value for white background
    threshold = 150  # Adjust this value based on your requirements

    # Check if the mean value is above the threshold
    if mean_value >= threshold:
        print("The background is white.")
        cv2.putText(image, 'The background is white.', (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
    else:
        print("The background is not white.") 
        cv2.putText(image, 'The background is not white.', (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        # return

    # #Detect white background in then image
    # # Set the threshold for white background detection
    # threshold = 100 # Adjust this value based on your requirements

    # # Count the number of pixels with intensity above the threshold
    # white_pixels = np.sum(gray > threshold)

    # # Calculate the percentage of white pixels in the image
    # total_pixels = gray.shape[0] * gray.shape[1]
    # white_percentage = (white_pixels / total_pixels) * 100
    # print(white_percentage)

    # # Define a threshold percentage for white background
    # threshold_percentage = 50  # Adjust this value based on your requirements

    # # Check if the white percentage is above the threshold
    # if white_percentage >= threshold_percentage:
    #     print("The background is white.")
    # else:
    #     print("The background is not white.")
    #     sys.exit()

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check if the detected face is within the shoulder region
        shoulder_threshold = 1.5  # Adjust this value based on your requirements
        shoulder_y = int(y + 1.5 * h)  # Assume the shoulder is 1.5 times below the face
        shoulder_h = int(shoulder_threshold * h)
        cv2.rectangle(image, (x, shoulder_y), (x+w, shoulder_y+shoulder_h), (255, 0, 0), 2)

        # Detect eyes within the face region
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        # Check if no eyes are detected within the face region
        if len(eyes) ==2:
            print("Eyes are front facing")
            cv2.putText(image, 'Eyes Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            print("Eyes are not detected")
            cv2.putText(image, 'No Eyes Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # return

    # Display the resulting image
    cv2.imshow('Detected Person', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        file_paths.append(file_path)

# print(file_paths)

for file_path in file_paths:
    check_portrait_image(file_path)
    
# check_portrait_image( __location__ + '/images'+'/img_good_11.jpg')