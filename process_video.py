import time
from interface import VideoInfo
import validation as Validations
import aws_download_video as AWS_VIDEO_DOWNLOADER
import cv2
import numpy as np
from keras.models import model_from_json
from datetime import timedelta
import json
from collections import Counter


video_link = ""
video_type = ""
AWS_S3 = "AWS_S3"
BUCKET_NAME = "emoiton-detection-poc"
DOWNLOAD_PREFIX = "videos/"
DESTINATION_FOLDER = "temp"
# EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
EMOTION_DICT = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad"}
interval_seconds = 5
start_time = 0
end_time = interval_seconds
emotions_data = []
current_time = 0
result_emotions = []
overall_emotion_counts = {emotion: 0 for emotion in EMOTION_DICT.values()}
dominant_emotion_summary = None
emotion_statistics_summary = None


def download_video():
  if video_type == AWS_S3:
    downloaded_video_result = AWS_VIDEO_DOWNLOADER.main(
      bucketName=BUCKET_NAME,
      prefix=DOWNLOAD_PREFIX,
      videoKey=video_link,
      destinationFolder=DESTINATION_FOLDER
    )
    print(downloaded_video_result)
    return downloaded_video_result
  else:
    raise ValueError("Right now we only allow AWS_S3 type videos.")


def process_and_save_emotions_frame_by_frame(cap, emotion_model):
  global current_time, emotions_data
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    # Load face detector.
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # Convert the frame to grayScale image.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces available in frame.
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
      # FOR_LOCAL
      # cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
      # Taking only face from grayScale image.
      roi_gray_frame = gray_frame[y:y + h, x:x + w]
      cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

      # Predict the emotions
      emotion_prediction = emotion_model.predict(cropped_img)
      maxindex = int(np.argmax(emotion_prediction))
      print('EmotionIndex : ', maxindex)
      # FOR_LOCAL
      # cv2.putText(frame, EMOTION_DICT[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

      # Record the emotion along with the timestamp
      current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
      emotions_data.append((current_time, EMOTION_DICT[maxindex]))

    # FOR_LOCAL
    # cv2.imshow('Emotion Detection', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break


def analyse_emotions_in_small_time_blocks():
  global start_time, end_time, current_time, overall_emotion_counts, interval_seconds, result_emotions, emotions_data
  while start_time < current_time:
    window_emotions = [emotion for time, emotion in emotions_data if start_time <= time <= end_time]

    if window_emotions:
      dominant_emotion = max(set(window_emotions), key=window_emotions.count)
      emotion_statistics = {emotion: f"{window_emotions.count(emotion) / len(window_emotions):.2f}" for emotion in EMOTION_DICT.values()}
      
      # Update overall emotion counts
      overall_emotion_counts[dominant_emotion] += 1
    else:
      dominant_emotion = "Neutral"
      emotion_statistics = {emotion: "0.00" for emotion in EMOTION_DICT.values()}

    result_emotions.append({
      "start": str(timedelta(seconds=start_time)),
      "end": str(timedelta(seconds=end_time)),
      "dominant_emotion": dominant_emotion,
      "statistics": emotion_statistics
    })

    start_time = end_time
    end_time += interval_seconds


def calculate_video_summary():
  global emotions_data
  # Count occurrences of each emotion using Counter
  emotion_counts = Counter(emotion for _, emotion in emotions_data)
  # Calculate the percentage for each emotion
  emotion_statistics_summary = {
    emotion: f"{(count / len(emotions_data) * 100.0):.2f}" if count > 0 else "0.00"
    for emotion, count in emotion_counts.items()
  }
  # Additional iteration to ensure all emotions are included
  for emotion_id, emotion_name in EMOTION_DICT.items():
    if emotion_name not in emotion_statistics_summary:
      emotion_statistics_summary[emotion_name] = "0.00"
  # Find the dominant emotion
  dominant_emotion_summary = max(emotion_counts, key=emotion_counts.get)

  return {
    "dominant_emotion_summary": dominant_emotion_summary,
    "emotion_statistics_summary": emotion_statistics_summary
  }


def generate_final_result_object():
  global result_emotions, dominant_emotion_summary, emotion_statistics_summary
  return {
    "breakdown": result_emotions,
    "summary": {
      "dominant_emotion": dominant_emotion_summary,
      "statistics": emotion_statistics_summary
    }
  }


def main(body: VideoInfo):
  global video_link, video_type, dominant_emotion_summary, emotion_statistics_summary

  # validate if we have all fields.
  Validations.validate_body(body)

  # store all data.
  video_link = body.link
  video_type = body.type

  try:
    video_location = None
    # -------------------------------- DOWNLOAD THE VIDEO -------------------------------- #
    start_time = time.time()
    downloaded_video_result = download_video()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Download ended. Time taken: {elapsed_time:.2f} seconds")
    video_location = downloaded_video_result.get("location")

    # -------------------------------- LOADING EMOTION MODEL -------------------------------- #
    json_file = open('model/emotion_model_40.json', 'r')        # load json and create model
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model_40.h5")     # load weights into new model
    print("Model Loaded from Disk")

    # -------------------------------- SET THE VIDEO SOURCE -------------------------------- #
    Validations.check_file_location(video_location)
    # cap = cv2.VideoCapture(video_location)
    cap = cv2.VideoCapture(video_location)
    
    # -------------------------------- PROCESS AND DETECT EMOTION IN VIDEO -------------------------------- #
    start_time = time.time()
    process_and_save_emotions_frame_by_frame(cap=cap, emotion_model=emotion_model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Video processed. Time taken: {elapsed_time:.2f} seconds")
    cap.release()
    # FOR_LOCAL
    # cv2.destroyAllWindows()

    # -------------------------------- ANALYZE EMOTIONS IN 5-SECOND WINDOWS -------------------------------- #
    analyse_emotions_in_small_time_blocks()

    # -------------------------------- CALCULATE SUMMARY INFORMATION -------------------------------- #
    summary = calculate_video_summary()
    dominant_emotion_summary, emotion_statistics_summary = summary.values()

    # -------------------------------- CREATE THE FINAL RESULT -------------------------------- #
    # json_result = json.dumps(generate_final_result_object(), indent=2)
    json_result = generate_final_result_object()

    print("Final Emotion Analysis Result:")
    print(json_result)

    return json_result

  except Exception as e:
    return {"error": "An error occurred", "detail": str(e)}