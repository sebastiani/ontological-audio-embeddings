import numpy as np
import librosa
import os
import pickle


class DataUtils(object):

    def __init__(self, preprocPath):
        self.preprocPath = preprocPath


    def saveToNPY(self, path, resample=16000):
        audioFiles = os.listdir(path)
        audioFiles = [os.path.join(path, x) for x in audioFiles if 'm4a' in x]

        for audio in audioFiles:
            x, sr = librosa.load(audio, sr=resample)
            filename = audio.split('/')[-1]
            filename = filename.split('.')[0]
            filename = os.path.join(self.preprocPath, filename+'.npy')
            np.save(filename, x)

    def pkl2CSV(self, pkl, classesIndex=1):
        with open(pkl, 'rb') as f:
            rows = pickle.load(f)
        labels = []
        with open('dataset.csv', 'r') as out:
            labels = []
            for row in rows:
                filename = os.path.join(self.preprocPath, 'audio_'+row['YTID']+'.npy')
                label = None
                i = row.keys()[-1]
                try:
                    label = row[i].split(',')[classesIndex]

                except IndexError:
                    label = row[i].split(',')[0]

                labels.append(label)

                line = ','.join([filename, label]) + '\n'
                out.write(line)
        unique = list(set(labels))
        num = len(unique)
        numclass = range(len(unique))
        label_dict = zip((unique, numclass))
        with open('label_dict.pkl', 'wb') as out2:
            pickle.dump((label_dict, num), out2)

        print("Finished writing csv file")

