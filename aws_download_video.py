import boto3
import os


def downloadFile(bucketName, objectKey, downloadPath):
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucketName, objectKey, downloadPath)
        return { "error": False }
    except Exception as e:
        print(f"Error while downloading video from S3. Here is the log : {e}")
        return { "error": True, "message": e }


def main(bucketName, prefix, videoKey, destinationFolder):
    try:
        if not os.path.exists(destinationFolder):
            os.makedirs(destinationFolder)
        downloadPath = os.path.join(destinationFolder, os.path.basename(videoKey))
        objectKey = prefix+videoKey

        downloadResult = downloadFile(bucketName=bucketName, objectKey=objectKey, downloadPath=downloadPath)
        if downloadResult.get("error"):
            raise KeyError("Error while downloading the video. Please check logs for more details.")
        print("Video downloaded at : ", downloadPath)
        
        return { "downloaded": True, "location": downloadPath }
    except Exception as e:
        raise e