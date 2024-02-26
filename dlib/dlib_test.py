#=====================================================================
#  Face Recognition with Dlib - Test #=====================================================================

import os
import shutil

# Function to recognize faces in test images and move them to corresponding folders
def recognize_faces(image_paths):
    print("=====================================================================")
    print("Execution Started for Testing.")
    print("=====================================================================")
    for image_path in image_paths:
        # Extract only the image name from the full path
        image_name = os.path.basename(image_path)

        # Check if the file is an image
        if not os.path.isfile(image_path):
            print(f"File '{image_name}' does not exist.")
            continue
        if not any(image_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            print(f"Skipping non-image file: '{image_name}'")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print("Error loading image:", image_name)
            continue

        # Resize the image to 300x300
        resized_img = resize_image(img)

        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        matched = False
        for face in faces:
            shape = predictor(gray, face)

            # Compute face descriptor
            try:
                face_descriptor = face_rec_model.compute_face_descriptor(resized_img, shape)
                face_descriptor = np.array(face_descriptor)
            except Exception as e:
                print(f"Error computing face descriptor for '{image_name}': {e}")
                continue

            # Ensure face_descriptor has the correct shape
            if face_descriptor.shape != (128,):
                print(f"Invalid face descriptor shape for '{image_name}': {face_descriptor.shape}")
                continue

            # Compare the face descriptor with the trained encodings
            distances = np.linalg.norm(encodings_pickle - face_descriptor, axis=1)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            recognized_name = names_pickle[min_distance_index]

            # If the distance is below a certain threshold, consider it a match
            threshold = 0.6
            if min_distance < threshold:
                matched = True
                # Create a folder with the recognized name under 'test_matched'
                matched_folder = os.path.join('C:/Users/Learning/udemy/Hackathon/dlib/dataset/test/test_matched', recognized_name)
                os.makedirs(matched_folder, exist_ok=True)
                # Move the image to the matched folder
                shutil.move(image_path, os.path.join(matched_folder, image_name))
                print(f"Test Image: {image_name} Matched with: {recognized_name} Distance: {1 - min_distance:.4f}. Moved to '{recognized_name}' folder.")
                break
        
        if not matched:
            # Move the image to the unmatched folder
            shutil.move(image_path, os.path.join('C:/Users/Learning/udemy/Hackathon/dlib/dataset/test/test_unmatched', image_name))
            print(f"Test Image: {image_name} Not Matched with any model. Moved to 'test_unmatched' folder.")
    
    print("=====================================================================")
    print("Execution Completed for Testing.")
    print("=====================================================================")

# Test the face recognition on multiple sample images
test_images_path = 'C:/Users/Learning/udemy/Hackathon/dlib/dataset/test/test/'
image_files = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)]
recognize_faces(image_files)