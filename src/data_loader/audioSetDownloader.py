import sys
sys.path.append('/home/akasha/projects/ontological_audio_embeddings/') # hack to add upper visibility to this file
import youtube_dl
import pickle
import os
import multiprocessing
import subprocess
import shlex

from multiprocessing.pool import ThreadPool



DOWNLOAD = False
CUTTING = True
class DataDownloader(object):
    def __init__(self, csvPath, dataFolder):
        self.baseURL = 'http://youtu.be/{vid_id}?start={start}&end={end}'
        self.csvPath = csvPath
        self.dataFolder = dataFolder
        self.ydlOpts = {
            'format': 'bestaudio/best',
            'outtmpl': dataFolder + 'testing/' + 'audio_%(id)s.%(ext)s',
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
                        lineDict[headers[3].strip()] = ','.join(line[3:])

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
                
        baseCommand = 'youtube-dl --extract-audio --audio-format m4a {url} -o "{output}"'
        badUrls = []
        totalVideos = len(rows)
        cmds = []
        for i, video in enumerate(rows):
            if i < 13770:
                print("skipping...")
                continue
            print('Downloading {index}/{total}'.format(index=i, total=totalVideos))
            if video['YTID'] in skip:
                print('alredy downloaded')
                continue
            audioName = "raw_"+ video['YTID'] + ".m4a"
            video['filename'] = os.path.join(dataFolder, audioName)
            
            audioURL = self.baseURL.format(vid_id=video['YTID'], start=video['start_seconds'], end=video['end_seconds'])
            print("Downloading ", audioURL)
            urls = []
            if os.path.isfile(dataFolder+'audio_%s.m4a'% (video['YTID'])):
                print('already downloaded, skipping')
                continue
            cmd = baseCommand.format(url=audioURL, output=self.dataFolder + 'testing/' + 'audio_%(id)s.%(ext)s')
            cmds.append(cmd)
        
        pool = ThreadPool(multiprocessing.cpu_count())
        results = []
        for cmd in cmds:
            results.append(pool.apply_async(call_proc, (cmd,)))

        pool.close()
        pool.join()
        for result in results:
            out, err = result.get()
            with open('download_error.txt', 'w') as f:
                f.write(err)
            print("out: {} err: {}".format(out, err))
            
        """
        #bar = Bar('Processing', max=len(urls))
        with youtube_dl.YoutubeDL(self.ydlOpts) as ydl:
            for i in pool.imap_unordered(ydl.download, urls):
                #bar.next()
                print(i)
        #bar.finish()
            #pool.map_async(ydl.download, urls)
            #output = [p.get() for p in results]

        print("download finished")

        
            try:
                    ydl.download([audioURL])
            except:
                print('Downloading failed for ', audioURL)
                #badUrls.append(audioURL)
        #with open('badUrls.pkl', 'wb') as f:
        #    pickle.dump(badUrls, f)
        """

    def cutAudioSegments(self):
        baseCommand = "ffmpeg -i {input} -ss {start} -to {end} -c copy {output}"

        rows = self.csvParse()
        outputFolder = '/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio/'
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
        outputFolder = '/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio/'
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


    def parallelCutAudio(self, path):
        command = "ffmpeg -i {input} -ss {start} -to {end} -c copy {output}"
        rows = self.csvParse()
        commands = []

        outputFolder = '/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio2/'
        for i, video in enumerate(rows):
            filename = 'audio_' + video['YTID'] + '.m4a'
            inputFile = os.path.join(self.dataFolder, filename)
            outputFile = os.path.join(outputFolder, filename)
            if not os.path.isfile(inputFile):
                continue
            else:
                cmd  = command.format(input=inputFile, start=video['start_seconds'], end=video['end_seconds'], output=outputFile)
                print(cmd)
                commands.append(cmd)
            
        
        pool = ThreadPool(multiprocessing.cpu_count())
        results = []
        for cmd in commands:
            results.append(pool.apply_async(call_proc, (cmd, )))

        pool.close()
        pool.join()
        for result in results:
            out, err = result.get()
            with open('output_error.txt', 'w') as f:
                f.write(err)
            print("out: {} err: {}".format(out, err))

def call_proc(cmd):
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

def download(v):
    subprocess.check_call([
        'youtube-dl',
        '--extract-audio', '--audio-format', 'm4a',
        '-o', v[0], '--', v[1]
    ])



if __name__ == '__main__':
    csvPath = '/home/akasha/projects/ontological_audio_embeddings/data/raw/balanced_train_segments.csv'
    dataFolder = "/home/akasha/projects/ontological_audio_embeddings/data/raw/raw_audio/"
    dataDownloader = DataDownloader(csvPath, dataFolder)
    if DOWNLOAD:
        dataDownloader.downloadAudioSegments("/home/akasha/projects/ontological_audio_embeddings/data/raw")
    if CUTTING:
        #dataDownloader.testCutAudio("/home/akasha/projects/ontological_audio_embeddings/data/raw/batch_sample")
        #dataDownloader.cutAudioSegments()
        dataDownloader.parallelCutAudio("/home/akasha/projects/ontological_audio_embeddings/data/intermediate/raw_audio/")
