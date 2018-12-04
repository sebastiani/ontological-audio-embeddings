import sys
sys.path.append('/home/akasha/projects/ontological-audio-embeddings/') # hack to add upper visibility to this file
import youtube_dl
import pickle
import os


DOWNLOAD = False
CUTTING = True
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
                fname = url.split('_')
                ytid = fname[-1].split('.')[0]
                skip.append(ytid)
                
        
        badUrls = []
        for i, video in enumerate(rows):
            if video['YTID'] in skip:
                print('alredy downloaded')
                continue
            audioName = "raw_"+ video['YTID'] + ".m4a"
            video['filename'] = os.path.join(dataFolder, audioName)
            newRows.append(video)
            audioURL = self.baseURL.format(vid_id=video['YTID'], start=video['start_seconds'], end=video['end_seconds'])
            print("Downloading ", audioURL)

            with youtube_dl.YoutubeDL(self.ydlOpts) as ydl:
                try:
                    ydl.download([audioURL])
                except:
                    print('Downloading failed for ', audioURL)
                    badUrls.append(audioURL)
        with open('badUrls.pkl', 'wb') as f:
            pickle.dump(badUrls, f)

    def cutAudioSegments(self):
        baseCommand = "ffmpeg -ss {start} -t {end} -i {input} {output}"

        rows = self.csvParse()
        outputFolder = '/home/akasha/projects/ontological-audio-embeddings/data/intermediate/raw_audio/'
        for i, video in enumerate(rows):
            filename = 'audio_' + video['YTID'] + '.m4a'
            inputFile = os.path.join(self.dataFolder, filename)
            outputFile = os.path.join(outputFolder, filename)
            print("Cutting ", inputFile)
            os.system(baseCommand.format(start=video['start_seconds'], end=video['end_seconds'], input=inputFile, output=outputFile))


if __name__ == '__main__':
    csvPath = '/home/akasha/projects/ontological-audio-embeddings/data/raw/balanced_train_segments.csv'
    dataFolder = "/home/akasha/projects/ontological-audio-embeddings/data/raw/raw_audio/"
    dataDownloader = DataDownloader(csvPath, dataFolder)
    if DOWNLOAD:
        dataDownloader.downloadAudioSegments("/home/akasha/projects/ontological-audio-embeddings/data/raw")
    if CUTTING:
        dataDownloader.cutAudioSegments()
