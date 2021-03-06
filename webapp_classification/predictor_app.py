import os
from allennlp.nn.util import move_to_device
import torch

from flask import Flask
from flask import render_template
from flask import request

import json

from SocialMediaIE.data.tokenization import get_match_iter, get_match_object

from SocialMediaIE.predictor.model_predictor_classification import run, get_args, PREFIX, get_model_output, output_to_json


app = Flask(__name__)

SERIALIZATION_DIR = os.path.realpath("../data/models_classification/all_multitask_shared_bilstm_l2_0_lr_1e-3/")
args = get_args(PREFIX, SERIALIZATION_DIR)
args = args._replace(cuda=False)
TASKS, vocab, model, readers, test_iterator = run(args)

def tokenize(text):
    objects = [get_match_object(match) for match in get_match_iter(text)]
    n = len(objects)
    cleaned_objects = []
    for i, obj in enumerate(objects):
        obj["no_space"] = True
        if obj["type"] == "space":
            continue
        if i < n-1 and objects[i+1]["type"] == "space":
            obj["no_space"] = False
        cleaned_objects.append(obj)
    keys = cleaned_objects[0].keys()
    final_sequences = {}
    for k in keys:
        final_sequences[k] = [obj[k] for obj in cleaned_objects]
    return final_sequences


@app.route("/", methods=["POST", "GET"])
def predict():
    # Empty cache to ensure larger batch can be loaded for testing
    if request.method == "POST":
        text = request.form['textInput']
        data = [tokenize(text)]
    else:
        text = "Barack Obama went to Paris and never returned to the USA."
        text1 = "Stan Lee was a legend who developed Spiderman and the Avengers movie series."
        text2 = "I just learned about donald drumph through john oliver. #JohnOliverShow such an awesome show."
        data = [text, text1, text2]
        data = [tokenize(text) for text in data]
    torch.cuda.empty_cache()
    tokens = [obj["value"] for obj in data]
    output = list(get_model_output(model, tokens, args, readers, vocab, test_iterator))
    idx = 0
    output_json = output_to_json(tokens[idx], output[idx], vocab)
    output_json["text"] = text
    return render_template(
        "classification.html",
        text=text,
        output_json=output_json
    )


@app.route("/predict_json", methods=["POST", "GET"])
def predict_json():
    # Empty cache to ensure larger batch can be loaded for testing
    if request.method == "POST":
        text = request.form['textInput']
        data = [tokenize(text)]
    else:
        text = "Barack Obama went to Paris and never returned to the USA."
        text1 = "Stan Lee was a legend who developed Spiderman and the Avengers movie series."
        text2 = "I just learned about donald drumph through john oliver. #JohnOliverShow such an awesome show."
        data = [text, text1, text2]
        data = [tokenize(text) for text in data]
    torch.cuda.empty_cache()
    tokens = [obj["value"] for obj in data]
    output = list(get_model_output(model, tokens, args, readers, vocab, test_iterator))
    idx = 0
    output_json = output_to_json(tokens[idx], output[idx], vocab)
    output_json["text"] = text
    return app.response_class(
        response=json.dumps(output_json, indent=2),
        status=200,
        mimetype='application/json'
    )

