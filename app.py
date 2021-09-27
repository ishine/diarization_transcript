from diaUtils import *
from flask import Flask, request ,jsonify
from diaUtils import *

app  = Flask(__name__)


@app.route("/predict", methods = ["POST"])
def predict():
    audio_file = request.files["file"]
    audio_file.save(audio_file.filename)
    output = diatran()
    transcript = output.predict(audio_file.filename)
    # response = Response(output, mimetype="text/csv")
    return jsonify(output) 

if __name__ == "__main__":
    app.run(debug= False)

