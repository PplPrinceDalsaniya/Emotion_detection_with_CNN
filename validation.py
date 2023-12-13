import os
from fastapi import HTTPException
from interface import VideoInfo


def validate_body(body: VideoInfo):
  if not body.link:
    raise HTTPException(status_code=400, detail="Invalid video Link")
  if not body.type:
    raise HTTPException(status_code=400, detail="Invalid video Type")
  

def check_file_location(location):
  # Check if location is None
  if location is None:
    raise ValueError("File Location does not exists.")

  # Check if the directory exists, and create it if it doesn't
  directory = os.path.dirname(location)
  if not os.path.exists(directory):
    raise FileNotFoundError(f"Directory does not exist: {directory}")

  # Check if the file exists
  if not os.path.isfile(location):
    raise FileNotFoundError(f"File does not exist: {location}")

  return True