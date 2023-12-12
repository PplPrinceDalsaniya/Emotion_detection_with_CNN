from fastapi import HTTPException
from interface import VideoInfo


def validate_body(body: VideoInfo):
  if not body.link:
    raise HTTPException(status_code=400, detail="Invalid video Link")
  if not body.type:
    raise HTTPException(status_code=400, detail="Invalid video Type")