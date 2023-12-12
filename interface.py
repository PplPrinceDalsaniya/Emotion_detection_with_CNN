from pydantic import BaseModel


class VideoInfo(BaseModel):
  link: str
  type: str