import sys
sys.path.append('/home/akasha/projects/ontological-audio-embeddings/') # hack to add upper visibility to this file
import youtube_dl
import pickle
import os
import threading


DOWNLOAD = True
CUTTING = False
class DataDownloader(object):
    def __init__(self, csvPath, dataFolder):
        self.baseURL = 'http://youtu.be/{vid_id}?start={start}&end={end}'
        self.csvPath = csvPath
        self.dataFolder = dataFolder
        self.ydlOpts = {
            'format': 'bestaudio/best',
            'outtmpl': dataFolder + 'audio_%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
        }

    def csvParse(self, savePkl=None):
        skip=2
        lineCount = 0
        rows = []
        with open(self.csvPath, 'r') as csvFile:
            for line in csvFile:

                if lineCount < skip:
                    lineCount += 1
                    continue
                else:
                    if lineCount == skip:

                        headers = line[1:].strip()
                        headers = headers.split(',')
                    else:
                        line = line.strip()
                        line = line.split(',')
                        lineDict = dict()
                        lineDict[headers[0].strip()] = line[0].strip()
                        lineDict[headers[1].strip()] = line[1].strip()
                        lineDict[headers[2].strip()] = line[2].strip()
                        lineDict[headers[3].strip()] = line[3].join(',')
                        rows.append(lineDict)
                lineCount += 1
            if savePkl:
                filename = os.path.join(savePkl, 'dataDictionary.pkl')
                with open(filename, 'wb') as out:
                    pickle.dump(rows, out)
        return rows

    def downloadAudioSegments(self, folder=None):
        rows = self.csvParse(self.dataFolder)
        newRows = []
        # implement a checkpoint to continue downloading later.
        skip = []
        with open('progress.txt', 'r') as f:
            for url in f:
                url = url.strip()
                fname = url.split('_')
                ytid = fname[-1].split('.')[0]
                skip.append(ytid)
                

        badUrls = []
        totalVideos = len(rows)
        for i, video in enumerate(rows):
            if i < 4779:
                print("skipping...")
                continue
            print('Downloading {index}/{total}'.format(index=i, total=totalVideos))
            if video['YTID'] in skip:
                print('alredy downloaded')
                continue
            audioName = "raw_"+ video['YTID'] + ".m4a"
            video['filename'] = os.path.join(dataFolder, audioName)
            newRows.append(video)
            audioURL = self.baseURL.format(vid_id=video['YTID'], start=video['start_seconds'], end=video['end_seconds'])
            print("Downloading ", audioURL)
            if os.path.isfile(dataFolder+'audio_%s.m4a'% (video['YTID'])):
                print('already downloaded, skipping')
                continue
            with youtube_dl.YoutubeDL(self.ydlOpts) as ydl:
                try:
                    ydl.download([audioURL])
                except:
                    print('Downloading failed for ', audioURL)
                    badUrls.append(audioURL)
        with open('badUrls.pkl', 'wb') as f:
            pickle.dump(badUrls, f)

    def cutAudioSegments(self):
        baseCommand = "ffmpeg -i {input} -ss {start} -to {end} -c copy {output}"

        rows = self.csvParse()
        outputFolder = '/home/akasha/projects/ontological-audio-embeddings/data/intermediate/raw_audio/'
        for i, video in enumerate(rows):
            filename = 'audio_' + video['YTID'] + '.m4a'
            inputFile = os.path.join(self.dataFolder, filename)
            outputFile = os.path.join(outputFolder, filename)
            print("Cutting ", inputFile)
            os.system(baseCommand.format(start=video['start_seconds'], end=video['end_seconds'], input=inputFile, output=outputFile))

    def testCutAudio(self, path):
        baseCommand = "ffmpeg -i {input} -ss {start} -to {end} -c copy {output}"

        rows = self.csvParse()
        mini_rows = []
        outputFolder = '/home/akasha/projects/ontological-audio-embeddings/data/intermediate/raw_audio/'
        for video in os.listdir(path):
            if 'm4a' in video:
                vid_meta = None
                ytid = video.strip().split('_')[-1]
                ytid = ytid.split('.')[0]
                for meta in rows:
                    if meta['YTID'] == ytid:
                        vid_meta = meta
                        mini_rows.append(meta)
                    else:
                        continue
                inputFile = os.path.join(path, video)
                outputFile = os.path.join(outputFolder, video)
                print("Cutting ", inputFile)
                os.system(baseCommand.format(start=vid_meta['start_seconds'], end=vid_meta['end_seconds'], input=inputFile,
                                             output=outputFile))

        with open('miniData.pkl', 'wb') as f:
            pickle.dump(mini_rows, f)



if __name__ == '__main__':
    csvPath = '/home/akasha/projects/ontological-audio-embeddings/data/raw/balanced_train_segments.csv'
    dataFolder = "/home/akasha/projects/ontological-audio-embeddings/data/raw/raw_audio/"
    dataDownloader = DataDownloader(csvPath, dataFolder)
    if DOWNLOAD:
        dataDownloader.downloadAudioSegments("/home/akasha/projects/ontological-audio-embeddings/data/raw")
    if CUTTING:
        dataDownloader.testCutAudio("/home/akasha/projects/ontological-audio-embeddings/data/raw/batch_sample")
        #dataDownloader.cutAudioSegments()
