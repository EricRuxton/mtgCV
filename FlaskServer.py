import os
import cv2
import OcrUtils

from flask import Flask, render_template, request, Response
from pytesseract import pytesseract


# ---------------------- Flask App Setup ----------------------
app = Flask(__name__)
pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")


# ---------------------- Routes ----------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ReadCard', methods=['POST'])
def ReadCard():
    if request.method == 'POST':
        img64 = request.form["base64Img"]
        readText, setName, img = OcrUtils.ReadTitleFromCard(img64)

        # For debugging
        print(readText)
        print("Set: " + setName)
        _, buffer = cv2.imencode('.jpg', img)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

        # return Response(f"Read Text: {readText}")

    return render_template('index.html')


# ---------------------- Image super resolution - unused ----------------------
'''
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
load_dotenv()


def PreprocessImage(img):
    Prepare image for TensorFlow super-resolution model.
    if img.shape[-1] == 4:
        img = img[..., :-1]
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)
'''

if __name__ == '__main__':
    app.run(debug=True)
