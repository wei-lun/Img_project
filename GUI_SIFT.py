import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from joblib import load

FIXED_FEATURE_LENGTH = 128 
CLASS_MAPPING = {0: "healthy", 1: "tip burn"}  

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  
    return img

def segment_image(model, image):
    img = np.expand_dims(image, axis=0)  
    segmented_img = model.predict(img)[0]  
    return (segmented_img * 255).astype('uint8') 

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is not None:
        if descriptors.shape[0] >= FIXED_FEATURE_LENGTH:
            feature_vector = descriptors[:FIXED_FEATURE_LENGTH].flatten()
        else:
            padded_descriptors = np.zeros((FIXED_FEATURE_LENGTH, descriptors.shape[1]))
            padded_descriptors[:descriptors.shape[0]] = descriptors
            feature_vector = padded_descriptors.flatten()
    else:
        feature_vector = np.zeros(FIXED_FEATURE_LENGTH * 128) 
    return feature_vector

def predict_image(image_path, unet_model, svm_model, knn_model):
    image = load_image(image_path)
    segmented_image = segment_image(unet_model, image)

    feature_vector = extract_sift_features(segmented_image)
    feature_vector = np.expand_dims(feature_vector, axis=0)  

    svm_prediction = svm_model.predict(feature_vector)[0]
    knn_prediction = knn_model.predict(feature_vector)[0]

    return segmented_image, CLASS_MAPPING[svm_prediction], CLASS_MAPPING[knn_prediction]

class MainWindow(QMainWindow):
    def __init__(self, unet_model, svm_model, knn_model):
        super().__init__()
        self.unet_model = unet_model
        self.svm_model = svm_model
        self.knn_model = knn_model

        self.image_label = QLabel("原圖", self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.segmented_label = QLabel("分割圖", self)
        self.segmented_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("預測結果", self)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.open_button = QPushButton("開啟圖片", self)
        self.open_button.clicked.connect(self.open_image)

        self.predict_svm_button = QPushButton("使用 SVM 預測", self)
        self.predict_svm_button.clicked.connect(self.predict_svm)

        self.predict_knn_button = QPushButton("使用 KNN 預測", self)
        self.predict_knn_button.clicked.connect(self.predict_knn)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.segmented_label)
        layout.addWidget(self.result_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.predict_svm_button)
        button_layout.addWidget(self.predict_knn_button)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_path = None
        self.segmented_image = None

    def open_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "開啟圖片", "", "Images (*.png *.xpm *.jpg)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
            self.result_label.setText("")
            self.original_image = cv2.imread(self.image_path)

    def predict_svm(self):
        if not self.image_path:
            self.result_label.setText("請先開啟圖片")
            return
        self.segmented_image, svm_result, _ = predict_image(self.image_path, self.unet_model, self.svm_model, self.knn_model)
        self.display_segmented_image()
        self.result_label.setText(f"SVM 預測: {svm_result}")

    def predict_knn(self):
        if not self.image_path:
            self.result_label.setText("請先開啟圖片")
            return
        self.segmented_image, _, knn_result = predict_image(self.image_path, self.unet_model, self.svm_model, self.knn_model)
        self.display_segmented_image()
        self.result_label.setText(f"KNN 預測: {knn_result}")


    def display_segmented_image(self):
        if self.segmented_image is not None:
            if len(self.segmented_image.shape) == 3:
                self.segmented_image = self.segmented_image[:, :, 0]

            _, mask = cv2.threshold(self.segmented_image, 127, 255, cv2.THRESH_BINARY)

            original_image = cv2.resize(self.original_image, (128, 128))  
            masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

            height, width, channel = masked_image.shape
            bytes_per_line = channel * width
            q_image = QImage(masked_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.segmented_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    unet_model = tf.keras.models.load_model('unet_model_lettuce_3.h5')
    svm_model = load('svm_classifier_SIFT.pkl')
    knn_model = load('knn_classifier_SIFT.pkl')

    main_window = MainWindow(unet_model, svm_model, knn_model)
    main_window.setWindowTitle("Tip Burn 預測系統")
    main_window.resize(600, 800)
    main_window.show()

    sys.exit(app.exec_())
