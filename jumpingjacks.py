import mediapipe as mp
import cv2 as cv
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

videoframe = cv.VideoCapture(0)

#Functions

def calculate_angle(a, b, c):

    angle_radians = np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x)

    angle_degrees = np.abs(angle_radians*180/np.pi)

    if(angle_degrees > 180.0):
        angle_degrees = 360-angle_degrees
    
    ref = tuple([b.x, b.y])

    ref_int = tuple(np.multiply(ref, (640, 480)).astype(int))

    return angle_degrees, ref_int


def calculate_hip_displacement(current_left, current_right, prev_left, prev_right):

    if current_left - prev_left > 0 and current_right - prev_right > 0:
        return True
    
    return False


def display(angle, ref, frame):

    if angle < 150.0:
        cv.putText(
            frame, "{} Not straight", (50, 50), 
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA
        )    

    cv.putText(frame, str(angle), ref, 
        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)


#Mainloop
while True:

    data, frame = videoframe.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = pose.process(frame)

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    if results:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness = 2, circle_radius = 3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness = 2)
        )

        if results.pose_landmarks:

            count = 0
            state = ""

            landmarks = results.pose_landmarks.landmark

            #print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

            """ angle, ref_point = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ) """

            angle_left_arm, ref_point_left_arm = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            )

            angle_right_arm, ref_point_right_arm = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            )

            angle_left_leg, ref_point_left_leg = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )

            angle_right_leg, ref_point_right_leg = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )

            display(angle_left_arm, ref_point_left_arm, frame)
            display(angle_right_arm, ref_point_right_arm, frame)

            display(angle_left_leg, ref_point_left_leg, frame)
            display(angle_right_leg, ref_point_right_leg, frame)

    cv.imshow("Output", frame)

    if(cv.waitKey(1) & 0xFF == ord('q')):
        break

videoframe.release()
cv.destroyAllWindows()
