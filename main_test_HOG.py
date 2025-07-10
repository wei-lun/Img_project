import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

def load_images_and_labels(dataset_path):
    images = []
    labels = []
    class_mapping = {"healthy": 0, "tip burn": 1}  

    for class_name, label in class_mapping.items():
        class_folder = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def segment_images(model, images):
    segmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)
        segmented_img = model.predict(img)[0]
        segmented_img = (segmented_img * 255).astype('uint8')
        segmented_img = np.squeeze(segmented_img)  
        segmented_images.append(segmented_img)
    print(f"Segmented images shape: {np.array(segmented_images).shape}")
    return np.array(segmented_images)

def extract_hog_features(images):
    feature_vectors = []
    for i, img in enumerate(images):
        print(f"Processing image {i}, shape: {img.shape}, dtype: {img.dtype}")
        if len(img.shape) == 3 and img.shape[2] == 3:  
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:  
            gray_image = img
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        features, hog_image = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            multichannel=False
        )
        feature_vectors.append(features)
    return np.array(feature_vectors)

def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved as {filename}.")

def load_model(filename):
    model = load(filename)
    print(f"Model loaded from {filename}.")
    return model

"""def predict_image(image_path, unet_model, svm_classifier, knn_classifier):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    img = np.expand_dims(img, axis=0)
    segmented_img = unet_model.predict(img)[0]
    segmented_img = (segmented_img * 255).astype('uint8')

    if len(segmented_img.shape) == 3: 
        gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = segmented_img

    feature_vector, _ = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        multichannel=False
    )
    feature_vector = np.array([feature_vector])  

    class_mapping = {0: "healthy", 1: "tip burn"}
    svm_result = class_mapping[svm_classifier.predict(feature_vector)[0]]
    knn_result = class_mapping[knn_classifier.predict(feature_vector)[0]]
    return svm_result, knn_result"""

if __name__ == "__main__":
    unet_model = tf.keras.models.load_model('unet_model_lettuce_3.h5')

    dataset_path = "E:/lettuce_dataset"
    images, labels = load_images_and_labels(dataset_path)

    segmented_images = segment_images(unet_model, images)

    feature_vectors = extract_hog_features(segmented_images)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    save_model(svm_classifier, 'svm_classifier_HOG.pkl')

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    save_model(knn_classifier, 'knn_classifier_HOG.pkl')

    svm_predictions = svm_classifier.predict(X_test)
    knn_predictions = knn_classifier.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_predictions) * 100:.2f}%")
    print(f"KNN Accuracy: {accuracy_score(y_test, knn_predictions) * 100:.2f}%")


