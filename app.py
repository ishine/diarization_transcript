# from diaUtils import *
from flask import Flask, request ,jsonify
# from diaUtils import *
import diaUtils2
app  = Flask(__name__)
@app.route('/', methods = ["GET"])
def home(): 
    return "<H1>Hellow</h1>"

@app.route("/predict", methods = ["POST"])
def predict():
    audio_file = request.files["file"]
    audio_file.save(audio_file.filename)
    output = diatran()
    transcript = output.predict(audio_file.filename)
    # response = Response(output, mimetype="text/csv")
    return (output) 

@app.route('/test', methods = ["POST"])
def test():
    fileName = 'chunk0_1.wav'
    return jsonify(diaUtils2.main(fileName))

if __name__ == "__main__":
    app.run(debug= False)

