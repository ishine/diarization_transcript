import diaUtils2
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import subprocess
import srt
import json
import datetime
import sys


WORDS_PER_LINE = 7

sample_rate=16000
model = Model("/home/milindsoni/Documents/wisepal_all_features/wisepal/Resemblyzer-master/python/example/model")


def transcribeprocess(wav_fpath):
    rec = KaldiRecognizer(model, sample_rate,"flipkart")
    rec.SetWords(True)
    # sys.argv[1],
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i', wav_fpath ,
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)
    # with open(wav_fpath.split('.')[0]+ '.txt', 'w') as f:
    #     f.write((srt.compose(transcribe(rec,process))))
    transsrt = srt.compose(transcribe(rec,process))

   
    #rec = KaldiRecognizer(model, 16000, "zero oh one two three four five six seven eight nine")
    return rec, process, transsrt




def transcribe(rec, process):
    results = []
    subs = []
    while True:
       data = process.stdout.read(4000)
       if len(data) == 0:
           break
       if rec.AcceptWaveform(data):
           results.append(rec.Result())
    results.append(rec.FinalResult())

    for i, res in enumerate(results):
       jres = json.loads(res)
       if not 'result' in jres:
           continue
       words = jres['result']
       for j in range(0, len(words), WORDS_PER_LINE):
           line = words[j : j + WORDS_PER_LINE] 
           s = srt.Subtitle(index=len(subs), 
                   content=" ".join([l['word'] for l in line]),
                   start=datetime.timedelta(seconds=line[0]['start']), 
                   end=datetime.timedelta(seconds=line[-1]['end']))
           subs.append(s)
    return subs



     
    











# def transcribe(wav_fpath):
#     results = []
#     subs = []
#     sample_rate=16000
#     model = Model("/home/milindsoni/Documents/wisepal_all_features/wisepal/Resemblyzer-master/python/example/model")
#     rec = KaldiRecognizer(model, sample_rate)
#     rec.SetWords(True)
#     # sys.argv[1],
#     WORDS_PER_LINE = 7

#     process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i', wav_fpath ,
#                                 '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
#                                 stdout=subprocess.PIPE)
#     while True:
#        data = process.stdout.read(4000)
#        if len(data) == 0:
#            break
#        if rec.AcceptWaveform(data):
#            results.append(rec.Result())
#     results.append(rec.FinalResult())

#     for i, res in enumerate(results):
#        jres = json.loads(res)
#        if not 'result' in jres:
#            continue
#        words = jres['result']
#        for j in range(0, len(words), WORDS_PER_LINE):
#            line = words[j : j + WORDS_PER_LINE] 
#            s = srt.Subtitle(index=len(subs), 
#                    content=" ".join([l['word'] for l in line]),
#                    start=datetime.timedelta(seconds=line[0]['start']), 
#                    end=datetime.timedelta(seconds=line[-1]['end']))
#            subs.append(s)
#     return subs

# def clubbing(mappings):
#     clubbed = []
#     starIndex = mappings[0][0]
#     i = 0

#     while i<len(mappings):
#         temp = mappings[i]
#         k = i+1
#         while k<len(mappings) and mappings[k][0]==starIndex:
#             k+=1

#         i = k

#         if k>=len(mappings): # explicit ending condition
#             k = len(mappings)-1
#             i = len(mappings)
#             temp[-1] = mappings[-1][-1]
#             clubbed.append(temp)
#             break

#         starIndex = mappings[k][0]
#         temp[-1] = mappings[k-1][-1]
#         clubbed.append(temp)
#     return clubbed

# def main(wavef_path):
#     ## default variables
#     SetLogLevel(-1)
#     sample_rate=16000
#     model = Model("/home/milindsoni/Documents/wisepal_all_features/wisepal/Resemblyzer-master/python/example/model")
#     rec = KaldiRecognizer(model, sample_rate)
#     rec.SetWords(True)
#     # sys.argv[1],
#     process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i', wav_fpath , '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'], stdout=subprocess.PIPE)

#     WORDS_PER_LINE = 7
#     with open(wav_fpath.split('.')[0]+ '.txt', 'w') as f:
#         f.write((srt.compose(transcribe())))
#     ## Logic to return sr.compose:
#     return array
# #    combinedLabellings = diaUtils.main(wavef_path)
# #    mappings = combinedLabellings
# #    clubbed = clubbing(mappings)

#     ## returns 

# ## to make structed data out of text/transcript
# def convertString2Time(string): 
    
#     s = string.replace(',', '.')
#     s = s.split(":")
#     # print(s,end = ' ')
#     mult = (60)**(len(s)-1)
#     for i in range(len(s)): 
#         s[i] = float(s[i])*mult
#         mult/=60
#     # print(s)
#     return sum(s)
# def script2List(text): 
#     sets = []
#     temp = []
#     for i in range(len(text)): 
        
#         if (i+1)%4==0: 
#             sets.append(temp)
#             temp = []
#             continue
#         temp.append(text[i])
    
#     # filter 1: to remove \n 
#     for i in range(len(sets)):
#         s = sets[i]
#         for j in range(3): 
#             sets[i][j] = sets[i][j].split('\n')[0]
#     # filter 2: to remove --> from time periods!
#     for i in range(len(sets)): 
#         sets[i][1] =  sets[i][1].split(' --> ')
#     for i in range(len(sets)):
#     temp = []
#     for j in range(2):
#         temp.append(convertString2Time(sets[i][1][j]))
#     sets[i][1] = temp

#     s = []
#     for i in sets:
#         temp = [i[0], i[1][0], i[1][1], i[2]]
#         s.append(temp)
#     sets = s
#     return sets



# def transmodel(wavef_path):
#     if not os.path.exists("model"):
#         print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
#         exit (1)
    
#     sample_rate=16000
#     model = Model("/home/milindsoni/Documents/wisepal_all_features/wisepal/Resemblyzer-master/python/example/model")
#     rec = KaldiRecognizer(model, sample_rate)
#     rec.SetWords(True)
#     # sys.argv[1],
#     process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i', wave_fpath,
#                                 '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
#                                 stdout=subprocess.PIPE)
    
    
#     WORDS_PER_LINE = 7
#     return process, rec
    
    
  
    


