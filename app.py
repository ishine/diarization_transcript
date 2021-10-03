# from diaUtils import *
from flask import Flask, request ,jsonify, Response
from pathlib import Path
import moviepy.editor as mp
import os
# from diaUtils import *
import diaUtils2
app  = Flask(__name__)
@app.route('/', methods = ["GET"])
def home(): 
    return "<H1>Hellow</h1>"

@app.route("/predict", methods = ["POST"])
def predict():
    vid_file = request.files["file"]
    vid_file.save(vid_file.filename)
    clip = mp.VideoFileClip(vid_file.filename)
    clip.audio.write_audiofile("audio/"+vid_file.filename.split('.')[0]+".wav")
    wav_fpath = os.path.join("audio/",vid_file.filename.split('.')[0]+".wav")

    return Response(diaUtils2.main(wav_fpath).to_json(), mimetype='application/json')

    # response = Response(output, mimetype="text/csv")

@app.route('/test', methods = ["POST"])
def test():
    VidnameName = 'test2.mp4'
    clip = mp.VideoFileClip(os.path.basename(VidnameName))
    clip.audio.write_audiofile("audio/"+os.path.basename(VidnameName).split('.')[0]+".wav")

    wav_fpath = os.path.join("audio/",os.path.basename(VidnameName).split('.')[0]+".wav")
    #return jsonify(diaUtils2.main(wav_fpath))
    return Response(diaUtils2.main(wav_fpath).to_json(), mimetype='application/json')
if __name__ == "__main__":
    app.run(debug= False)

