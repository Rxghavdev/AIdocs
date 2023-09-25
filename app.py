from flask import Flask, request, jsonify, render_template
import tensorflow as tf 
from keras.models import load_model
from PIL import Image
from io import BytesIO
from utils import cleanImage
import pickle
import numpy as np
import gdown

file_url = "https://drive.google.com/uc?id=1iy5CWEtMou_XHowgV2Qy3YQctdN7_3Zh"

model_filename = "mask_detection.h5"

# Download the file
gdown.download(file_url, model_filename, quiet=False)

# Load the model
mask_detection_model = load_model(model_filename)
blur_detection_model = load_model("./ML/blur_sharp_classifier.h5") # blur => 0, sharp => 1
beard_detection_model = load_model("./ML/beard_detector.h5") # beard => 0, no_beard => 1 

# sentiment analysis
sentiment_analysis_model = load_model("./ML/sentiment_model.h5")
with open("./ML/tokenizer.pickle", "rb") as token_file:
    sentimentAnalysisTokenizer = pickle.load(token_file)

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


# sentiment analysis
@app.route("/sentiment")
def sentimeForm():
    return render_template('sentiment.html')

@app.route("/sentiment", methods=["POST"])
def sentimentAnalysis():
    text = request.form['text']

    input_sequence = sentimentAnalysisTokenizer.texts_to_sequences([text])

    # Perform custom padding
    max_length = 100  # Define the maximum sequence length
    if len(input_sequence[0]) < max_length:
        # If the sequence is shorter than max_length, pad it with 0s at the end
        padded_input = input_sequence[0] + [0] * (max_length - len(input_sequence[0]))
    else:
        # If the sequence is longer than max_length, truncate it
        padded_input = input_sequence[0][:max_length]

    padded_input = np.array([padded_input]).reshape(1, -1)
    predictions = sentiment_analysis_model.predict(padded_input)
    predicted_sentiment_index = np.argmax(predictions[0])
    output = ["positive", "negative", "neutral"][predicted_sentiment_index]

    return render_template('output.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)
