import mediapipe as mp
import cv2 as cv
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

videoframe = cv.VideoCapture(0)

#Variables
left_hip_prev = 0
right_hip_prev = 0
count = 0
didJump = False
up = None
down = None

#Functions

def calculate_angle(a, b, c):

    angle_radians = np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x)

    angle_degrees = np.abs(angle_radians*180/np.pi)

    if(angle_degrees > 180.0):
        angle_degrees = 360-angle_degrees
    
    ref = tuple([b.x, b.y])

    ref_int = tuple(np.multiply(ref, (640, 480)).astype(int))

    return angle_degrees, ref_int


def calculate_hip_displacement(left, right, prev_left, prev_right):

    global left_hip_prev, right_hip_prev, up, down

    print("Prev", prev_left, prev_right)
    current_left = left.y
    current_right = right.y
    print("Current", current_left, current_right)

    l_disp = current_left - prev_left
    r_disp = current_right - prev_right
    print("Disp", l_disp, r_disp)

    left_hip_prev = current_left
    right_hip_prev = current_right

    if l_disp > 0 and r_disp > 0:
        up = True
        down = False
        return True
    
    if l_disp < 0 and r_disp < 0:
        up = False
        down = True


def display(angle, ref, frame):

    """ if angle < 150.0:
        cv.putText(
            frame, "{} Not straight", (250, 50), 
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA
        )    """ 

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

            landmarks = results.pose_landmarks.landmark

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

            angle_hip_leftknee, ref_point_hipleftknee = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            )

            angle_hip_rightknee, ref_point_hiprightknee = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            )

            angle_shoulder_leftarm, ref_point_shoulder_leftarm = calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            )

            angle_shoulder_rightarm, ref_point_shoulder_rightarm = calculate_angle(
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            )

            display(angle_left_arm, ref_point_left_arm, frame)
            display(angle_right_arm, ref_point_right_arm, frame)

            display(angle_left_leg, ref_point_left_leg, frame)
            display(angle_right_leg, ref_point_right_leg, frame)

            display(angle_hip_leftknee, ref_point_hipleftknee, frame)
            display(angle_hip_rightknee, ref_point_hiprightknee, frame)

            display(angle_shoulder_leftarm, ref_point_shoulder_leftarm, frame)
            display(angle_shoulder_rightarm, ref_point_shoulder_rightarm, frame)

            calculate_hip_displacement(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                left_hip_prev, right_hip_prev
            )

            if up and \
            angle_left_arm > 150.0 and angle_right_arm > 150.0 \
            and angle_left_leg > 150.0 and angle_right_leg > 150.0 \
            and angle_hip_leftknee > 90.0 and angle_hip_rightknee > 90.0 \
            and angle_shoulder_leftarm > 170.0 and angle_shoulder_rightarm > 170.0:
                didJump = True
                cv.putText(frame, "Up", (40, 20), 
        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)
            
            if down and didJump \
            and angle_hip_leftknee < 95.0 and angle_hip_rightknee < 95.0 \
            and angle_shoulder_leftarm < 50.0 and angle_shoulder_rightarm < 50.0:
                count += 1
                didJump = False
                cv.putText(frame, "Down", (40, 20), 
        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)
            
    cv.putText(
        frame, "Count: " + str(count), (20, 100), 
        cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
        2, cv.LINE_AA
    )

    cv.imshow("Output", frame)

    if(cv.waitKey(1) & 0xFF == ord('q')):
        break

videoframe.release()
cv.destroyAllWindows()
