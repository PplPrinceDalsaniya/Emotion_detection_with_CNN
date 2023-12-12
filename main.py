from fastapi import FastAPI
import random
from interface import VideoInfo
import process_video as ProcessVideo


app = FastAPI()


@app.get('/')
async def root():
  return {"example": "This is example", "data": [1,2,3]}


# BODY : 
# {
#     "link": "https://example.com/sample_video.mp4",
#     "type": "AWS"
# }
@app.post("/process_video", status_code=200)
async def process_video(body: VideoInfo):
  return ProcessVideo.main(body)


@app.get('/random')
async def get_random():
  rn: int = random.randint(0, 100)
  return {
    'number': rn,
    'limit': 100
  }


@app.get('/random/{limit}')
async def get_random(limit: int):
  rn: int = random.randint(0, limit)
  return {
    'number': rn,
    'limit': limit
  }
