from flask import Flask, request, jsonify, render_template
import tensorflow as tf 
from keras.models import load_model
from PIL import Image
from io import BytesIO
from utils import cleanImage

# loading the models
mask_detection_model = load_model("./ML/mask_detection.h5") # mask => 0, no_mask => 1
blur_detection_model = load_model("./ML/blur_sharp_classifier.h5") # blur => 0, sharp => 1
beard_detection_model = load_model("./ML/beard_detector.h5") # beard => 0, no_beard => 1 

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

# mask detectionn
@app.route('/mask', methods=["GET"])
def maskForm():
    return render_template('mask.html')

@app.route('/mask', methods=["POST"])
def maskDetection():
    if 'image' not in request.files:
        return render_template('output.html', output="No file part")

    file = request.files['image']

    if file:
        img = Image.open(BytesIO(file.read()))
        img_array = cleanImage(img)
        output = mask_detection_model.predict(img_array)[0][0]

        if output >= 0.5:
            output = "No Mask"
        else:
            output = "Mask"

        return render_template('output.html', output=output)
    else:
        return render_template('output.html', output="No file found")


# blur/sharp detection
@app.route("/blur")
def blurForm():
    return render_template('blur.html')

@app.route("/blur", methods=["POST"])
def blurDetection():
    if 'image' not in request.files:
        return render_template('output.html', output="No file part")

    file = request.files['image']

    if file:
        img = Image.open(BytesIO(file.read()))
        img_array = cleanImage(img, 224, 224)
        output = blur_detection_model.predict(img_array)[0][0]

        if output >= 0.5:
            output = "Sharp"
        else:
            output = "Blur"

        return render_template('output.html', output=output)
    else:
        return render_template('output.html', output="No file found")


# beard detection
@app.route("/beard")
def beardForm():
    return render_template('beard.html')

@app.route("/beard", methods=["POST"])
def beardDetection():
    if 'image' not in request.files:
        return render_template('output.html', output="No file part")

    file = request.files['image']

    if file:
        img = Image.open(BytesIO(file.read()))
        img_array = cleanImage(img, 224, 224)
        output = beard_detection_model.predict(img_array)[0][0]

        if output >= 0.5:
            output = "No Beard"
        else:
            output = "Beard"

        return render_template('output.html', output=output)
    else:
        return render_template('output.html', output="No file found")


if __name__ == '__main__':
    app.run(debug=True)
