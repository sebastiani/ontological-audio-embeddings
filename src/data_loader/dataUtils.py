import numpy as np
import librosa
import os
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool


class DataUtils(object):

    def __init__(self, preprocPath):
        self.preprocPath = preprocPath


    def saveToNPY(self, path, resample=16000):
        audioFiles = os.listdir(path)
        audioFiles = [os.path.join(path, x) for x in audioFiles if 'm4a' in x]
        total = len(audioFiles)

        def saver(audio):
            x, sr = librosa.load(audio, sr=resample)
            filename = audio.split('/')[-1]
            filename = filename.split('.')[0]
            filename = os.path.join(self.preprocPath, filename + '.npy')
            np.save(filename, x)

        pool = ThreadPool(multiprocessing.cpu_count())

        results = []
        for i, audio in enumerate(audioFiles):
            results.append(pool.apply_async(saver, (audio,)))

        pool.close()
        pool.join()
        for result in results:
            out, err = result.get()
            print("out: {} err: {}".format(out, err))

        print("Finished converting to npy, resampled at {sampling} Hz".format(sampling=resample))

    def pkl2CSV(self, pkl, classesIndex=0):
        with open(pkl, 'rb') as f:
            rows = pickle.load(f)
        labels = []
        with open('dataset.csv', 'r') as out:
            labels = []
            for row in rows:
                filename = os.path.join(self.preprocPath, 'audio_'+row['YTID']+'.npy')
                if not os.path.isfile(filename):
                    continue
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


if __name__ == '__main__':
    utils = DataUtils('/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSet')
    intermediate = "/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio"
    utils.saveToNPY(intermediate)
    pkl = "/home/akasha/projects/ontological_audio_embeddings/data/raw/dataDictionary.pkl"
    utils.pkl2CSV(pkl)

