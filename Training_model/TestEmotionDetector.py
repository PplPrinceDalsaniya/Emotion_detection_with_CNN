import cv2
import numpy as np
from keras.models import model_from_json
from datetime import timedelta
import json

# -------------------------------- EMOTION DICTIONARY -------------------------------- #
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad"}

# -------------------------------- LOADING EMOTION MODEL -------------------------------- #
# load json and create model
json_file = open('model/emotion_model_20.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
# load weights into new model
emotion_model.load_weights("model/emotion_model_20.h5")
print("Loaded model from disk")

# -------------------------------- GLOBAL VARIABLES -------------------------------- #
# Time interval for emotion analysis
interval_seconds = 5
start_time = 0
end_time = interval_seconds
emotions_data = []

# -------------------------------- SET THE VIDEO SOURCE -------------------------------- #
# start the webcam feed
# cap = cv2.VideoCapture(0)

# pass here your video path
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/anger_closeUP.mp4")
cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/anger_girl.mp4")
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/Angry_boy.mp4")
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/FearAndSurprice.mp4")
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/Happy_girl.mp4")
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/sad_uncle.mp4")
# cap = cv2.VideoCapture("/Users/princedalsaniyappl/Downloads/Emotion Detection Downloads/Test_videos/test_video_1.mov")

# -------------------------------- PROCESS AND DETECT EMOTION IN VIDEO -------------------------------- #
emotions_data = []
current_time = 0

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        print('EmotionIndex : ', maxindex)
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Record the emotion along with the timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        emotions_data.append((current_time, emotion_dict[maxindex]))

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------- ANALYZE EMOTIONS IN 5-SECOND WINDOWS -------------------------------- #
interval_seconds = 5
start_time = 0
end_time = interval_seconds
result_emotions = []
overall_emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}

while start_time < current_time:
    window_emotions = [emotion for time, emotion in emotions_data if start_time <= time <= end_time]

    if window_emotions:
        dominant_emotion = max(set(window_emotions), key=window_emotions.count)
        emotion_statistics = {emotion: f"{window_emotions.count(emotion) / len(window_emotions):.2f}" for emotion in emotion_dict.values()}
        
        # Update overall emotion counts
        overall_emotion_counts[dominant_emotion] += 1
    else:
        dominant_emotion = "Neutral"
        emotion_statistics = {emotion: "0.00" for emotion in emotion_dict.values()}

    result_emotions.append({
        "start": str(timedelta(seconds=start_time)),
        "end": str(timedelta(seconds=end_time)),
        "dominant_emotion": dominant_emotion,
        "statistics": emotion_statistics
    })

    start_time = end_time
    end_time += interval_seconds

# -------------------------------- CALCULATE SUMMARY INFORMATION -------------------------------- #
total_windows = len(result_emotions)
dominant_emotion_summary = max(overall_emotion_counts, key=overall_emotion_counts.get)
emotion_statistics_summary = {emotion: f"{overall_emotion_counts[emotion] / total_windows:.2f}" for emotion in emotion_dict.values()}

# -------------------------------- PRINT THE FINAL RESULT -------------------------------- #
final_result = {
    "breakdown": result_emotions,
    "summary": {
        "dominant_emotion": dominant_emotion_summary,
        "statistics": emotion_statistics_summary
    }
}

# Convert the final result to a JSON-formatted string
json_result = json.dumps(final_result, indent=2)

print("Final Emotion Analysis Result:")
print(json_result)