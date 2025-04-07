from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib
from skimage.feature import hog, local_binary_pattern
from PIL import Image
import io

app = Flask(__name__)

# Load models once
scaler = joblib.load("scaler.pkl")
lda = joblib.load("lda.pkl")
model = joblib.load("ensemble_model.pkl")
le = joblib.load("label_encoder.pkl")

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys')
    hist_feat = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256]*3).flatten()

    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    lbp_feat = lbp_hist.astype("float")
    lbp_feat /= (lbp_feat.sum() + 1e-6)

    return np.concatenate([hog_feat.ravel(), hist_feat.ravel(), lbp_feat.ravel()])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    disease_name = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_bytes = file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if img is not None:
                resized = cv2.resize(img, (128, 128))
                features = extract_features(resized)
                features = scaler.transform(features.reshape(1, -1))
                features = lda.transform(features)
                pred = model.predict(features)[0]
                disease_name = le.inverse_transform([pred])[0]
                uploaded_image = image_bytes

    return render_template("index.html", prediction=disease_name, image=uploaded_image)

if __name__ == "__main__":
    app.run(debug=True)
