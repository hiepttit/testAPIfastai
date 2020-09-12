from flask import Flask, request, jsonify
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner('export.pkl')
classes = learn.dls.vocab


def predict_single(img_file):
    'function to take image and return prediction'
    prediction = learn.predict(img_file)
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()
