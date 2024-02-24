#!/usr/bin/env python
# coding: utf-8

# In[ ]:


fr_train_udpated


# In[ ]:


C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train/
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_cropped
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_processed
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_error


C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test/
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test_cropped
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test/test_processed
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test/test_error

C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/output
C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models



# # updated train code working copy

# In[2]:


import cv2
import numpy as np
import os
import face_recognition
import time
import h5py
import pickle
import shutil

# Source and destination paths
training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train/'
cropped_training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_cropped/'
pickle_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.pickle" # train model
h5_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.h5" # train model

# Define the desired sizes
initial_image_size = (300, 300)
cropped_image_size = (100, 100)

print("=====================================================================")
print("Execution Started.")
print("=====================================================================")

# Function to detect and resize faces
def detect_and_resize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None

    # Resize the image
    resized_image = cv2.resize(img, initial_image_size)

    return img, resized_image

# Function to detect and crop faces from resized image
def detect_and_crop_faces(resized_image):
    face_locations = face_recognition.face_locations(resized_image)

    cropped_faces = []
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = resized_image[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, cropped_image_size)

            cropped_faces.append(face_image_resized)

    return cropped_faces

# Function to create subdirectories
def create_subdirectories(source_path, dest_path):
    subdirs = [subdir for subdir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, subdir))]
    for subdir in subdirs:
        os.makedirs(os.path.join(dest_path, subdir), exist_ok=True)

# Function to move cropped images to corresponding subdirectories
def move_cropped_images(source_path, dest_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)
            person_name = os.path.basename(root)
            dest_dir = os.path.join(dest_path, person_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_file, dest_dir)

# Function to display cropped images
def display_cropped_images(cropped_faces):
    for i, cropped_face in enumerate(cropped_faces):
        cv2.imshow(f"Cropped Face {i+1}", cropped_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create subdirectories in training_cropped
create_subdirectories(training_path, cropped_training_path)

# Process images, detect faces, crop faces, and save cropped images
for root, _, files in os.walk(training_path):
    for file in files:
        image_path = os.path.join(root, file)
        try:
            _, resized_image = detect_and_resize_faces(image_path)

            if resized_image is not None:
                cropped_faces = detect_and_crop_faces(resized_image)

                if cropped_faces:
                    #display_cropped_images(cropped_faces)

                    for i, cropped_face in enumerate(cropped_faces):
                        person_name = os.path.basename(root)
                        cv2.imwrite(os.path.join(cropped_training_path, person_name, f"{file[:-4]}_{i}.jpg"), cropped_face)

        except Exception as e:
            print("Error processing image:", image_path)
            print(e)

# Encoding faces from cropped images
list_encodings = []
list_names = []

for root, _, files in os.walk(cropped_training_path):
    for file in files:
        image_path = os.path.join(root, file)
        name = os.path.basename(root)
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img_rgb)
            if encoding:
                list_encodings.append(encoding[0])
                list_names.append(name)
        except Exception as e:
            print("Error encoding image:", image_path)
            print(e)

# Save the encodings and names in a pickle file
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
    pickle.dump(encodings_data, f)

# Save the encodings and names in an H5 file
with h5py.File(h5_filename, "w") as hf:
    hf.create_dataset("encodings", data=np.array(list_encodings))
    hf.create_dataset("names", data=np.array(list_names, dtype='S'))

print("=====================================================================")
print("Execution completed.")
print("=====================================================================")


# In[ ]:


#


# In[3]:


import pickle

# Path to the pickle file containing encodings
pickle_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.pickle"

# Load encodings from the pickle file
with open(pickle_filename, "rb") as f:
    encodings_data = pickle.load(f)

# Extract encodings and names
encodings = encodings_data["encodings"]
names = encodings_data["names"]

# Print encodings
for name, encoding in zip(names, encodings):
    print("Name:", name)
    print("Encoding:", encoding)
    print()


# # Test  working copy

# In[12]:


import cv2
import numpy as np
import os
import face_recognition
import h5py
import pickle

# Function to load encodings and names from H5 file
def load_encodings_from_h5(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        encodings = hf["encodings"][:]
        names = hf["names"][:]
    return encodings, names

# Function to load encodings and names from pickle file
def load_encodings_from_pickle(pickle_filename):
    with open(pickle_filename, "rb") as f:
        data = pickle.load(f)
    encodings = data["encodings"]
    names = data["names"]
    return encodings, names
# Function to recognize faces in test images using encodings
def recognize_faces_in_test_images(encodings, names, test_images_folder):
    results = {}
    for root, _, files in os.walk(test_images_folder):
        for file in files:
            image_path = os.path.join(root, file)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = names[first_match_index]
                    confidence = face_recognition.face_distance(encodings, face_encoding)[first_match_index]
                    print("Match found using model:", name, "with confidence:", confidence, "for test image:", file)
                    # Move the matched file to the processed folder
                    matched_folder = os.path.join(test_images_folder, "test_matched", name)
                    os.makedirs(matched_folder, exist_ok=True)
                    shutil.move(image_path, os.path.join(matched_folder, file))
                else:
                    # Move the unmatched file to the unmatched folder
                    unmatched_folder = os.path.join(test_images_folder, "test_unmatched")
                    os.makedirs(unmatched_folder, exist_ok=True)
                    shutil.move(image_path, os.path.join(unmatched_folder, file))
                results[file] = name
    return results



# Function to compare face recognition results from both sources
def compare_face_recognition_results(encodings_h5, names_h5, encodings_pickle, names_pickle, test_images_folder):
    # Recognize faces in test images using encodings from H5 file
    results_h5 = recognize_faces_in_test_images(encodings_h5, names_h5, test_images_folder)
    
    # Recognize faces in test images using encodings from pickle file
    results_pickle = recognize_faces_in_test_images(encodings_pickle, names_pickle, test_images_folder)
    
    # Compare results and calculate accuracy
    total_images = len(results_h5)
    correct_h5 = sum(1 for image, name in results_h5.items() if results_pickle[image] == name)
    accuracy_h5 = (correct_h5 / total_images) * 100
    
    total_images_pickle = len(results_pickle)
    correct_pickle = sum(1 for image, name in results_pickle.items() if results_h5[image] == name)
    accuracy_pickle = (correct_pickle / total_images_pickle) * 100
    
    print("Accuracy using encodings from H5 file: {:.2f}%".format(accuracy_h5))
    print("Accuracy using encodings from pickle file: {:.2f}%".format(accuracy_pickle))

# Print execution started
print("Execution Started.")
print("=====================================================================")

# Load encodings and names from H5 file
encodings_h5, names_h5 = load_encodings_from_h5("C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.h5")

# Load encodings and names from pickle file
encodings_pickle, names_pickle = load_encodings_from_pickle("C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.pickle")

# Compare face recognition results from both sources
compare_face_recognition_results(encodings_h5, names_h5, encodings_pickle, names_pickle,"C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test/")


# Print execution ended
print("=====================================================================")
print("Execution completed.")


# # Test Final working copy

# In[26]:


import cv2
import numpy as np
import os
import face_recognition
import time
import h5py
import pickle
import shutil

# Source and destination paths
training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train/'
cropped_training_path = 'C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/train/train_cropped/'
pickle_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.pickle" # train model
h5_filename = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/trained_models/face_encodings_custom.h5" # train model
test_images_folder = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test/"
test_matched_folder = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test_matched/"
test_unmatched_folder = "C:/Users/Learning/udemy/Hackathon/ultimate/fr_updated/dataset/test/test_unmatched/"

# Define the desired sizes
initial_image_size = (300, 300)
cropped_image_size = (100, 100)

print("=====================================================================")
print("Execution Started.")
print("=====================================================================")

# Function to detect and resize faces
def detect_and_resize_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image:", image_path)
        return None

    # Resize the image
    resized_image = cv2.resize(img, initial_image_size)

    return img, resized_image

# Function to detect and crop faces from resized image
def detect_and_crop_faces(resized_image):
    face_locations = face_recognition.face_locations(resized_image)

    cropped_faces = []
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = resized_image[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, cropped_image_size)

            cropped_faces.append(face_image_resized)

    return cropped_faces

# Function to create subdirectories
def create_subdirectories(source_path, dest_path):
    subdirs = [subdir for subdir in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, subdir))]
    for subdir in subdirs:
        os.makedirs(os.path.join(dest_path, subdir), exist_ok=True)

# Create subdirectories for cropped training images
create_subdirectories(training_path, cropped_training_path)

# Function to move cropped images to corresponding subdirectories
def move_cropped_images(source_path, dest_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            src_file = os.path.join(root, file)
            person_name = os.path.basename(root)
            dest_dir = os.path.join(dest_path, person_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_file, dest_dir)

# Process images, detect faces, crop faces, and save cropped images
for root, _, files in os.walk(training_path):
    for file in files:
        image_path = os.path.join(root, file)
        try:
            _, resized_image = detect_and_resize_faces(image_path)

            if resized_image is not None:
                cropped_faces = detect_and_crop_faces(resized_image)

                if cropped_faces:
                    for i, cropped_face in enumerate(cropped_faces):
                        person_name = os.path.basename(root)
                        cv2.imwrite(os.path.join(cropped_training_path, person_name, f"{file[:-4]}_{i}.jpg"), cropped_face)

        except Exception as e:
            print("Error processing image:", image_path)
            print(e)

# Encoding faces from cropped images
list_encodings = []
list_names = []

for root, _, files in os.walk(cropped_training_path):
    for file in files:
        image_path = os.path.join(root, file)
        name = os.path.basename(root)
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img_rgb)
            if encoding:
                list_encodings.append(encoding[0])
                list_names.append(name)
        except Exception as e:
            print("Error encoding image:", image_path)
            print(e)

# Save the encodings and names in a pickle file
encodings_data = {"encodings": list_encodings, "names": list_names}
with open(pickle_filename, "wb") as f:
    pickle.dump(encodings_data, f)

# Save the encodings and names in an H5 file
with h5py.File(h5_filename, "w") as hf:
    hf.create_dataset("encodings", data=np.array(list_encodings))
    hf.create_dataset("names", data=np.array(list_names, dtype='S'))



# Function to recognize faces in test images using encodings
def recognize_faces_in_test_images(encodings, names, test_images_folder):
    results = {}
    for root, _, files in os.walk(test_images_folder):
        for file in files:
            image_path = os.path.join(root, file)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = names[first_match_index]
                    confidence = face_recognition.face_distance(encodings, face_encoding)[first_match_index]
                    print("Exact match found:", name.decode("utf-8"), "with confidence:", confidence, "for test image:", file)
                    
                    # Create subfolder based on the name# Convert byte strings to regular strings
                    name_str = name.decode('utf-8')
                    # Remove leading 'b' and quotes
                    name_str = name_str[2:-1] if name_str.startswith("b'") else name_str
                    
                    matched_folder = os.path.join(os.path.dirname(os.path.dirname(test_images_folder)), "test_matched")

                    # Create folder without 'b' and quotes
                    test_matched_folder = os.path.join(matched_folder, name_str)
                    # Create the destination directory if it doesn't exist
                    os.makedirs(test_matched_folder, exist_ok=True)

                    #matched_folder = os.path.join(test_matched_folder, name)
                    #matched_folder = os.path.join(str(test_matched_folder), str(name))
                    # Move the matched file to the matched folder
                    shutil.move(image_path, os.path.join(matched_folder, file))
                else:
                    # Move the unmatched file to the unmatched folder
                    test_unmatched_folder = os.path.join(os.path.dirname(os.path.dirname(test_images_folder)), "test_unmatched")
                    os.makedirs(test_unmatched_folder, exist_ok=True)
                    shutil.move(image_path, os.path.join(test_unmatched_folder, file))
                results[file] = name
    return results


# Call the function to recognize faces in test images using encodings
recognize_faces_in_test_images(encodings_h5, names_h5, test_images_folder)

print("=====================================================================")
print("Execution completed.")
print("=====================================================================")


# In[ ]:




