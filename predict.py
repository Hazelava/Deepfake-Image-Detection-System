import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("deepfake_detector.h5")

# Load test image
img = cv2.imread("test.jpg")  # put your test image here
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))

# Predict
prediction = model.predict(img)

if prediction > 0.5:
    print("Fake Image")
else:
    print("Real Image")