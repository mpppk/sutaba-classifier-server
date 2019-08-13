import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from keras import models

app = Flask(__name__)
model = models.load_model('sutaba-model.h5')
model.summary()
graph = tf.get_default_graph()

classes = [
    'sutaba',
    'ramen',
    'other',
]


def crop_center_as_maximized_square(pil_img):
    img_width, img_height = pil_img.size
    c = min(img_width, img_height)
    return crop_center(pil_img, c, c)


def crop_center_as_square(pil_img, crop):
    img_width, img_height = pil_img.size
    c = min(img_width, img_height, crop)
    return crop_center(pil_img, c, c)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size

    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'file' in request.files:
        img = request.files['file'].read()
        img = Image.open(io.BytesIO(img))
        img = crop_center_as_maximized_square(img)
        img_size = int(os.environ.get('KERAS_MODEL_INPUT_SIZE', 224))
        img = img.resize((img_size, img_size))
        img = np.asarray(img) / 255.
        img = np.expand_dims(img, axis=0)
        global graph
        with graph.as_default():
            pred = model.predict(img)
            confidence = str(round(max(pred[0]), 3))
            pred = classes[np.argmax(pred)]

            data = dict(pred=pred, confidence=confidence)
            return jsonify(data)
    return '{error: "image does not exist in file key"}'


if __name__ == '__main__':
    # load_model()
    app.run(debug=True, host="0.0.0.0", port=os.environ.get('PORT', 5000))
