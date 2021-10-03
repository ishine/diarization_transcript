## user-defined module
from utils import *
# in-built modules
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import json
from pydub.utils import make_chunks
from scipy.spatial.distance import pdist, squareform, euclidean, cosine
from spectralcluster import SpectralClusterer
import moviepy.editor as mp
from resemblyzer import sampling_rate
import pandas as pd
from trans import *
import subprocess
import io
import pandas as pd
from flask import Flask, request ,jsonify, Response

filename = 'chunk0_1.wav'   

wav_fpath = Path(filename)
encoder = VoiceEncoder("cpu")
clusterer = SpectralClusterer(
            min_clusters=2,
            max_clusters=100,
            #p_percentile=0.90,
            #gaussian_blur_sigma=1
            )

if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)


def get_large_audio_transcription(wav_fpath):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """    
    # open the audio file using pydub
    sound = AudioSegment.from_wav(wav_fpath)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunk_length_ms = 120000 # pydub calculates in millisec
    chunks = make_chunks(sound, chunk_length_ms) #M
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_json = ""
    labelling = []
    global_label = []
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        print('start preprocessing:' + chunk_filename)
        # recognize the chunk
        #wav = preprocess_wav(wav_fpath)
        wav = preprocess_wav(chunk_filename)
        print('preprocessing done:' + chunk_filename)
        #wav = audio_chunk
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
        labels = clusterer.predict(cont_embeds)
        times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
        
        start_time = 0
        print('create labels:' + chunk_filename)
#         print(times)
        perChunkLabel = []
        for j,time in enumerate(times):
            if j>0 and labels[j]!=labels[j-1]:
                temp = [str(labels[j-1]),start_time,time]
                perChunkLabel.append(temp)
                start_time = time
            if j==len(times)-1:
                temp = [str(labels[j]),start_time,time]
                perChunkLabel.append(temp)
#         chunkl.append(len(labelling))   #positions of all the chunks in 
        labelling.append(perChunkLabel)
    return labelling, chunks, wav


def processlabel(global_label,wav):
    if global_label[0][-1][0] == '1':
        last_1_time = global_label[0][-1][1:]
    else: 
        last_1_time = global_label[0][-2][1:]
    
    embed1_time = wav[int(last_1_time[0]*sampling_rate):int(last_1_time[1]*sampling_rate)]

    constant = global_label[0][-1][2]
    new_global = global_label.copy()
    shouldChange = []
    new_global.pop(0)
    k=0
    for i in new_global: 
        global start_i, end_i
        start_i, end_i= i[1][1], i[1][2]
        print(start_i)
        embedi_time = wav[int((start_i+global_label[k][-1][2])*1000):int((end_i+global_label[k][-1][2])*1000)]
        k += 1 
        if get_similarity(encoder.embed_utterance(embed1_time), encoder.embed_utterance(embedi_time)): 
            if i[-1][0] == '0': 
                changeBits(i)
                shouldChange.append(True)
            else: 
                shouldChange.append(False)
        else: 
            if i[-1][0] == '0': 
                shouldChange.append(False)
            else: 
                shouldChange.append(True)
                changeBits(i)

    new_global.insert(0,global_label[0])
    
    noOfFiles = len(new_global)
    combinedLabellings = joinlabels(new_global,noOfFiles)
    # print(combinedLabellings)

    for i in combinedLabellings:
        if(i[2]-i[1]<1):
            combinedLabellings.remove(i)
        return combinedLabellings
    ## returns are remaining
  
    return combinedLabellings


def clubbing(mappings):
    clubbed = []
    starIndex = mappings[0][0]
    i = 0

    while i<len(mappings):
        temp = mappings[i] 
        k = i+1
        while k<len(mappings) and mappings[k][0]==starIndex: 
            k+=1

        i = k

        if k>=len(mappings): # explicit ending condition
            k = len(mappings)-1
            i = len(mappings)
            temp[-1] = mappings[-1][-1]
            clubbed.append(temp)
            break

        starIndex = mappings[k][0]
        temp[-1] = mappings[k-1][-1]
        clubbed.append(temp)
    return clubbed

def combinetranscript(transcript,clubbed):
    buf = io.StringIO(transcript)
    text = buf.readlines()
    sets = []
    temp = []
    for i in range(len(text)): 
        
        if (i+1)%4==0: 
            sets.append(temp)
            temp = []
            continue
        temp.append(text[i])

    # filter 1: to remove \n 
    for i in range(len(sets)):
        s = sets[i]
        for j in range(3): 
            sets[i][j] = sets[i][j].split('\n')[0]
    # filter 2: to remove --> from time periods!
    for i in range(len(sets)): 
        sets[i][1] =  sets[i][1].split(' --> ')

   
    # To expand each array!
    for i in range(len(sets)): 
        temp = []
        for j in range(2):
            temp.append(convertString2Time(sets[i][1][j]))
        sets[i][1] = temp

    s = [] 
    for i in sets:
        temp = [i[0], i[1][0], i[1][1], i[2]]
        s.append(temp)
    sets = s

    df = pd.DataFrame(sets, columns = ['index', 'start', 'end', 'sentence'])
    df = df.set_index('index')
    
        
    
    k = 0
    npdf = df.values
    labels = []
    for i in range(len(npdf)): 
        if k>=len(clubbed): 
            labels.append(clubbed[-1][2])
            continue
            
        if npdf[i][1]<clubbed[k][2]:
            labels.append(clubbed[k][0])
        elif k<len(clubbed): 
            labels.append(clubbed[k][0])
            k+=1
    df['labels'] = labels
    #df.to_csv('withLabels.csv', index = False)

    return df

def convertString2Time(string): 
    
    s = string.replace(',', '.')
    s = s.split(":")
    # print(s,end = ' ')
    mult = (60)**(len(s)-1)
    for i in range(len(s)): 
        s[i] = float(s[i])*mult
        mult/=60
    # print(s)
    return sum(s)



def main(wav_fpath):    
    global_label, chunk, wav = get_large_audio_transcription(wav_fpath)
    combinedlabelling = processlabel(global_label,wav)
    clubbed = clubbing(combinedlabelling)
    transcript = transcribeprocess(wav_fpath)[2]
    df = combinetranscript(transcript,clubbed)
    # return Response(df.to_json(), mimetype='application/json')    # print(df)
    return df.to_dict()
if __name__ == "__main__":
    main()
 

