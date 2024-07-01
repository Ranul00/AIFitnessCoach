import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import datetime
import traceback
import tensorflow as tf
from tensorflow import keras 

import pickle

from shared.common_func import (
    calculate_angle,
    extract_important_keypoints,
    get_drawing_color,
    rescale_frame
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Determine important landmarks for plank
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

# Generate all columns of the data frame

HEADERS = ["label"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

class ShoulderPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None

    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
                             landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].visibility]

        is_visible = all(
            [vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible

        # Get joints' coordinates
        self.shoulder = [landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
                         landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y]
        self.elbow = [landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
                      landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y]
        self.wrist = [landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
                      landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y]
        self.hip = [landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].x,
                    landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].y]

        return self.is_visible


    def analyze_pose(self, landmarks, frame):
        '''
        - Front Raise Counter
        - Errors Detection
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate raise angle for counter
        front_raise_angle = int(calculate_angle(self.elbow, self.shoulder, self.hip))
        if front_raise_angle > self.stage_down_threshold:
            self.stage = "down"
        elif front_raise_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        # * Evaluation for LOOSE UPPER ARM error
        if front_raise_angle > self.loose_upper_arm_angle_threshold:
            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False

        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and front_raise_angle <= self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = front_raise_angle
            self.peak_contraction_frame = frame

        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                self.detected_errors["PEAK_CONTRACTION"] += 1

            # Reset params
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None

        return (front_raise_angle, None)

# Load input scaler
with open("./models/front_raise/input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)

DL_model = keras.models.load_model('./models/front_raise/hybrid_model_Corrected.h5')

# cap = cv2.VideoCapture(0)

VISIBILITY_THRESHOLD = 0.65

# Params for counter
STAGE_UP_THRESHOLD = 95
STAGE_DOWN_THRESHOLD = 30

# Params to catch FULL RANGE OF MOTION error
PEAK_CONTRACTION_THRESHOLD = 95

# LOOSE UPPER ARM error detection
LOOSE_UPPER_ARM = False
LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 99

# STANDING POSTURE error detection
POSTURE_ERROR_THRESHOLD = 0.95


# Init analysis class
left_arm_analysis = ShoulderPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

right_arm_analysis = ShoulderPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

def front_raise_detection(cap, image, cv2):
    posture = 0
    # Convert the frame to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # while cap.isOpened():
        #     ret, image = cap.read()

        #     if not ret:
        #         break

            # Reduce size of a frame
            image = rescale_frame(image, 50)
            image = cv2.flip(image, 1)
            video_dimensions = [image.shape[1], image.shape[0]]

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                print("No human found")
                # continue

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Make detection
            try:
                landmarks = results.pose_landmarks.landmark
                
                (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row, ], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                prediction = DL_model.predict(X)
                predicted_class = np.argmax(prediction, axis=1)[0]
                prediction_probability = round(max(prediction.tolist()[0]), 2)

                if prediction_probability >= POSTURE_ERROR_THRESHOLD:
                    posture = predicted_class

                print("test prob:  ", predicted_class, prediction_probability) 

                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (600, 40), (245, 117, 16), -1)

                # Display probability
                cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Left Counter
                cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # * Display error
                # Right arm error
                cv2.putText(image, "R_PC", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Left arm error
                cv2.putText(image, "L_PC", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (295, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Lean back error
                cv2.putText(image, "LB", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str("Correct" if posture == 0 else "Stable Your Hip") + f" ,{predicted_class}, {prediction_probability}", (350, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # * Visualize angles
                # Visualize LEFT arm calculated angles
                if left_arm_analysis.is_visible:
                    cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


                # Visualize RIGHT arm calculated angles
                if right_arm_analysis.is_visible:
                    cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            except Exception as e:
                print(f"Error: {e}")
            
            # cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("quit")

    return image
        #         break

        # cap.release()
        # cv2.destroyAllWindows()

        # # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
        # for i in range (1, 5):
        #     cv2.waitKey(1)
    
