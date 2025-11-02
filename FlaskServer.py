import os
import OcrUtils

from flask import Flask, render_template, request, Response
import cv2
import tensorflow as tf
from dotenv import load_dotenv
from pytesseract import pytesseract


# ---------------------- Flask App Setup ----------------------
app = Flask(__name__)
scale = 0.02450502473611342250789329380908  # inches / pixel scale
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
load_dotenv()
pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")


# ---------------------- Routes ----------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ReadCard', methods=['POST'])
def ReadCard():
    if request.method == 'POST':
        img64 = request.form["base64Img"]
        readText, img = OcrUtils.ReadTitleFromCard(img64)
        # return f"Read Text: {readText}"
        print(readText)
        _, buffer = cv2.imencode('.jpg', img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return render_template('index.html')


# ---------------------- Image super resolution - unused ----------------------
def PreprocessImage(img):
    """Prepare image for TensorFlow super-resolution model."""
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


if __name__ == '__main__':
    app.run(debug=True)
