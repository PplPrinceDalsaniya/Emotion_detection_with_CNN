from interface import VideoInfo
import validation as validation
import aws_download_video as AWS_VIDEO_DOWNLOADER


video_link = ""
video_type = ""
AWS_S3 = "AWS_S3"
BUCKET_NAME = "emoiton-detection-poc"
DOWNLOAD_PREFIX = "videos/"
DESTINATION_FOLDER = "temp"


def main(body: VideoInfo):
  global video_link, video_type

  # validate if we have all fields.
  validation.validate_body(body)

  # store all data.
  video_link = body.link
  video_type = body.type

  try:
    if video_type == AWS_S3:
      downloaded_video_result = AWS_VIDEO_DOWNLOADER.main(
        bucketName=BUCKET_NAME,
        prefix=DOWNLOAD_PREFIX,
        videoKey=video_link,
        destinationFolder=DESTINATION_FOLDER
      )
      return downloaded_video_result
    else:
      return {"error": "Right now we only allow AWS_S3 type videos."}

  except Exception as e:
    return {"error": "An error occurred", "detail": str(e)}