from interface import VideoInfo
import validation as validation


video_link = ""
video_type = ""


def main(body: VideoInfo):
  global video_link, video_type

  # validate if we have all fields.
  validation.validate_body(body)

  # store all data.
  video_link = body.link
  video_type = body.type

  return {"video_link": video_link, "type": video_type}