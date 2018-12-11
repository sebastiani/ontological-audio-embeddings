import sys
sys.path.append('/home/akasha/projects/')
import numpy as np
import librosa
import os
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool

from ontological_audio_embeddings.src.data_loader.audioSetDownloader import DataDownloader


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
        with open('dataset.csv', 'w') as out:
            labels = []
            for row in rows:
                filename = os.path.join(self.preprocPath, 'audio_'+row['YTID']+'.npy')
                if not os.path.isfile(filename):
                    continue
                label = None
                i = list(row.keys())[-1]
                try:
                    label = row[i].split(',')[classesIndex]

                except IndexError:
                    label = row[i].split(',')[0]

                labels.append(label.strip().replace('"', ''))

                line = ','.join([filename, label.strip().replace('"', '')]) + '\n'
                out.write(line)
        unique = list(set(labels))
        num = len(unique)
        numclass = range(len(unique))
        label_dict = dict(zip(unique, numclass))
        with open('label_dict.pkl', 'wb') as out2:
            pickle.dump((label_dict, num), out2)

        print("Finished writing csv file")


    def padOrTruncate(self, source, dest):
        files = os.listdir(source)

        def pad_truncate(filename):
            x = np.load(os.path.join(source, filename))
            desired_len = 160086
            out = os.path.join(dest, filename)
            if x.shape[0] == desired_len:
                np.save(out, x)

            elif x.shape[0] < desired_len:
                padw = desired_len - x.shape[0]
                x = np.pad(x, (0, padw), mode='constant')
                np.save(out, x)

            else:
                x = x[:desired_len]
                np.save(out, x)

        pool = ThreadPool(multiprocessing.cpu_count())
        results = []
        for file in files:
            results.append(pool.apply_async(pad_truncate, (file,)))

        pool.close()
        pool.join()

        print("Finished!")

    def normalize(self, source, dest):
        files = os.listdir(source)

        maxes = []
        mins = []
        for f in files:
            if '.npy' in f:
                filename = os.path.join(source, f)
                x = np.load(filename)
                m = max(x)
                maxes.append(m)
                m = min(x)
                mins.append(m)

        f_max = max(maxes)
        f_min = min(mins)

        m = 2.0/(f_max - f_min)
        b = -1.0 - m*f_min
        def normalize(f):
            filename = os.path.join(source, f)
            x = np.load(filename)
            new_x = m*x + b
            new_filename = os.path.join(dest, f)
            np.save(new_filename, new_x)

        pool = ThreadPool(multiprocessing.cpu_count())
        results = []
        for file in files:
            results.append(pool.apply_async(normalize, (file, )))

        pool.close()
        pool.join()




if __name__ == '__main__':
    csvPath = '/home/akasha/projects/ontological_audio_embeddings/data/raw/balanced_train_segments.csv'
    dataFolder = "/home/akasha/projects/ontological_audio_embeddings/data/raw/raw_audio/"
    dataDownloader = DataDownloader(csvPath, dataFolder)

    #rows = dataDownloader.csvParse("/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/")
    utils = DataUtils('/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSet')
    #intermediate = "/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio"
    #utils.saveToNPY(intermediate)
    #pkl = "/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/dataDictionary.pkl"
    #utils.pkl2CSV(pkl)
    #utils.padOrTruncate('/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSet',
    #                    '/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSetv2')
    utils.normalize('/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSetv2',
                         '/home/akasha/projects/ontological_audio_embeddings/data/preprocessed/rawAudioSetv3')

