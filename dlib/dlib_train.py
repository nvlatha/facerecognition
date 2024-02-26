#=====================================================================
# Face Recognition with dlib - Train 
#====================================================================="

import cv2
import numpy as np
import os
import dlib
import h5py
import pickle
import shutil
from sklearn.decomposition import PCA

# Source and destination paths
training_path = 'C:/Users/Learning/udemy/Hackathon/dlib/dataset/train/train/'
cropped_training_path = 'C:/Users/Learning/udemy/Hackathon/dlib/dataset/train/train_cropped/'
pickle_filename = "C:/Users/Learning/udemy/Hackathon/dlib/dataset/trained_models/fe_dlib_custom.pickle" # train model
h5_filename = "C:/Users/Learning/udemy/Hackathon/dlib/dataset/trained_models/fe_dlib_custom.h5" # train model
shape_predictor_path = "C:/Users/Learning/udemy/Hackathon/dlib/dataset/trained_models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "C:/Users/Learning/udemy/Hackathon/dlib/dataset/trained_models/dlib_face_recognition_resnet_model_v1.dat"

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
# Initialize face recognition model
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Define the desired sizes
initial_image_size = (300, 300)
cropped_image_size = (150, 150)

print("=====================================================================")
print("Execution Started for Training.")
print("=====================================================================")

# Function to display image
def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect and resize faces
def detect_and_resize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None, None

    if img.size == 0:
        print("Error: Empty image loaded:", image_path)
        return None, None

    # Resize the image to 300x300
    resized_image = cv2.resize(img, initial_image_size)

    # Display the resized image
    #display_image('Resized Image', resized_image)

    return img, resized_image

# Function to detect and crop faces from original image with landmarks
def detect_and_crop_faces_with_landmarks(original_image):
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_rects = detector(gray, 1)

    cropped_faces = []
    if face_rects:
        for face_rect in face_rects:
            # Use shape predictor to get precise facial landmarks
            shape = predictor(gray, face_rect)
            points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # Draw landmarks on the original image
            for (x, y) in points:
                cv2.circle(original_image, (x, y), 2, (0, 255, 0), -1)

            # Calculate the coordinates for cropping
            left = face_rect.left()
            top = face_rect.top()
            right = face_rect.right()
            bottom = face_rect.bottom()

            # Crop the face region from the original image
            face_image = original_image[top:bottom, left:right]

            # Resize the cropped face to 150x150
            face_image_resized = cv2.resize(face_image, (150, 150))

            cropped_faces.append((face_image_resized, (left, top, right, bottom)))

    return cropped_faces, original_image  # Return cropped faces and original image with landmarks drawn

# Function to create subdirectories
def create_subdirectories(source_path, dest_path):
    subdirs = [subdir for subdir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, subdir))]
    for subdir in subdirs:
        os.makedirs(os.path.join(dest_path, subdir), exist_ok=True)

# Create subdirectories in cropped_training_path
create_subdirectories(training_path, cropped_training_path)

# Encoding faces from cropped images
list_encodings = []
list_names = []

# Process images, detect faces, crop faces, and save cropped images
for root, _, files in os.walk(training_path):
    for file in files:
        image_path = os.path.join(root, file)
        try:
            original_image, resized_image = detect_and_resize_faces(image_path)

            if original_image is not None and resized_image is not None:
                cropped_faces_info, marked_image = detect_and_crop_faces_with_landmarks(original_image)

                if cropped_faces_info:
                    for i, (cropped_face, face_coordinates) in enumerate(cropped_faces_info):
                        person_name = os.path.basename(root)
                        # Save cropped images in the corresponding subfolder
                        subfolder_path = os.path.join(cropped_training_path, person_name)
                        os.makedirs(subfolder_path, exist_ok=True)
                        cv2.imwrite(os.path.join(subfolder_path, f"{file[:-4]}_{i}.jpg"), cropped_face)

                        # Encode the face
                        img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                        face_descriptor = face_rec_model.compute_face_descriptor(img_rgb)
                        encoding = np.array(face_descriptor)

                        if encoding is not None:
                            list_encodings.append(encoding)
                            list_names.append(person_name)

                        # Display the marked image with landmarks
                        #display_image('Marked Image with Landmarks', marked_image)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()

        except Exception as e:
            print("Error processing image:", image_path)
            print(e)

# Save the encodings and names in a pickle file
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
    pickle.dump(encodings_data, f)

# Save the encodings and names in an H5 file
if list_encodings:
    encodings_array = np.array(list_encodings)
    # Check if the array is empty
    if encodings_array.size == 0:
        raise ValueError("Array is empty. No samples found.")
    else:
        # Initialize PCA with the number of components
        n_components = min(encodings_array.shape[0], encodings_array.shape[1])
        pca = PCA(n_components=n_components)
        flattened_encodings = pca.fit_transform(encodings_array.reshape(-1, 128))

        # Save the encodings and names in an H5 file
        with h5py.File(h5_filename, "w") as hf:
            hf.create_dataset("encodings", data=flattened_encodings)
            hf.create_dataset("names", data=np.array(list_names, dtype='S'))

print("=====================================================================")
print("Execution Completed for Training.")
print("=====================================================================")