# importing libraries 
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

def changeBits(chunkLabel): 
    for i in range(len(chunkLabel)):
        if chunkLabel[i][0] =='0': 
            chunkLabel[i][0] = '1'
        else: 
            chunkLabel[i][0] = '0'
    return chunkLabel


def get_similarity(embed_i, embed_j):
    if (1-cosine(embed_i, embed_j)) >= 0.8 :
        return 1
    else:
        return 0
def joinlabels(orderedLabelling,noOfFiles):
    combinedLabellings = []
    secondsCount= 0
    for i in range(noOfFiles):
        tempArray = orderedLabelling[i]
        isSameAsLastSpeaker=False
        
        if (len(combinedLabellings) == 0):
            combinedLabellings.extend(tempArray)
            
        else:
            if (combinedLabellings[len(combinedLabellings)-1][0] == tempArray[0][0]):
                combinedLabellings[len(combinedLabellings)-1][2] = combinedLabellings[len(combinedLabellings)-1][2]+ tempArray[0][2]
                isSameAsLastSpeaker=True
            for j in range(len(tempArray)):
                if (j > 0 or isSameAsLastSpeaker == False):
                    tempArray[j][1] = tempArray[j][1] + secondsCount # 135.5*(i)
                    tempArray[j][2] = tempArray[j][2] +secondsCount # 135.5*(i)
                    combinedLabellings.append(tempArray[j])
        print( combinedLabellings[len(combinedLabellings)-1][2])
        secondsCount = combinedLabellings[len(combinedLabellings)-1][2]
    return combinedLabellings

#def createLabellings(filepath, noOfFiles):
#    for i in range(noOfFiles):
#        with open(filepath +'/interview' + str(i) + '.json', 'r') as data_file:
#            json_data = data_file.read()
#        samplelabellings.append(json.loads(json_data))


## from transcription: 

