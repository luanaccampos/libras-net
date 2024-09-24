from pytube import YouTube, Search
import pickle

d = {}

def Download(link, out):
    try:
        youtubeObject = YouTube(link)
        youtubeObject = youtubeObject.streams.get_highest_resolution()
        youtubeObject.download(output_path='libras-videos', filename=f'{out}.mp4')
        print("Download is completed successfully")
    except:
        print("An error has occurred")
    

s = Search('libras alfabeto')
s.results
s.get_next_results()

print(f'{len(s.results)} videos')

for x in s.results:
  if x.length < 360 and x.video_id not in d:
    d[x.video_id] = x.watch_url
    Download(x.watch_url, x.video_id)
    
with open('metadata-videos.pkl', 'wb') as f:
    pickle.dump(d, f)