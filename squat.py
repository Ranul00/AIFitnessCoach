import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import traceback
import tensorflow as tf
from tensorflow import keras 
import math


import pickle

from shared.common_func import (
    calculate_angle,
    extract_important_keypoints,
    get_drawing_color,
    rescale_frame
)


# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for Bicep Curl
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# Generate all columns of the data frame
headers = ["label"] # Label column

for lm in IMPORTANT_LMS:
    headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


def calculate_distance(pointX, pointY) -> float:
    '''
    Calculate a distance between 2 points
    '''

    x1, y1 = pointX
    x2, y2 = pointY

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    '''
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis

    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    '''
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results

    # * Calculate shoulder width
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)

    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2

    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        print("Cannot see knee")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)

    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get(
        "up")
    middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get(
        "middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get(
        "down")

    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
            
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
            
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
            
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
            
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
            
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
            
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
            
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
            
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
            

    # print(analyzed_results)
    return analyzed_results


# Load model for counter
with open("./models/squat/LR_model.pkl", "rb") as f:
    count_model = pickle.load(f)




PREDICTION_PROB_THRESHOLD = 0.7

# Error vars
VISIBILITY_THRESHOLD = 0.6
FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": [0.5, 1.0],
    "middle": [0.7, 1.0],
    "down": [0.7, 1.1],
}

def squat_detection(cap, image, cv2): 
    current_stage = ""
    # Counter vars
    counter = 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # while cap.isOpened():
        #     ret, image = cap.read()

        #     if not ret:
        #         break

            # Reduce size of a frame
            image = rescale_frame(image, 50)


            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                # continue
                print("No human found")

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
                244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Make detection
            try:

                # * Model prediction for SQUAT counter
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row], columns=headers[1:])

                # Make prediction and its probability
                predicted_class = count_model.predict(X)[0]
                prediction_probabilities = count_model.predict_proba(X)[0]
                prediction_probability = round(prediction_probabilities[prediction_probabilities.argmax()], 2)

                # Evaluate model prediction
                
                if predicted_class == 1 and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                    current_stage = "down"
                    print("inside down if")
                elif current_stage == "down" and predicted_class == 0 and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                    current_stage = "up"
                    print("inside up if")
                    counter += 1


                # Analyze squat pose
                analyzed_results = analyze_foot_knee_placement(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS,
                                                            knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)

                foot_placement_evaluation = analyzed_results["foot_placement"]
                knee_placement_evaluation = analyzed_results["knee_placement"]

                # * Evaluate FOOT PLACEMENT error
                if foot_placement_evaluation == -1:
                    foot_placement = "UNK"
                elif foot_placement_evaluation == 0:
                    foot_placement = "Correct"
                elif foot_placement_evaluation == 1:
                    foot_placement = "Too tight"
                elif foot_placement_evaluation == 2:
                    foot_placement = "Too wide"

                # * Evaluate KNEE PLACEMENT error
                if knee_placement_evaluation == -1:
                    knee_placement = "UNK"
                elif knee_placement_evaluation == 0:
                    knee_placement = "Correct"
                elif knee_placement_evaluation == 1:
                    knee_placement = "Too tight"
                elif knee_placement_evaluation == 2:
                    knee_placement = "Too wide"

                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

                str_pedicted_class = str(predicted_class)

                # Display class
                cv2.putText(image, "COUNT", (10, 12),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f'{str(counter)}, {str_pedicted_class.split(" ")[0]}, {str(prediction_probability)}', (
                    5, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Foot and Shoulder width ratio
                cv2.putText(image, "FOOT", (200, 12),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, foot_placement, (195, 40),
                            cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display knee and Shoulder width ratio
                cv2.putText(image, "KNEE", (330, 12),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, knee_placement, (325, 40),
                            cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
            
            # cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("quit")

    return image
        # cap.release()
        # cv2.destroyAllWindows()

        # # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
        # for i in range (1, 5):
        #     cv2.waitKey(1)