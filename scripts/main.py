import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None


def calculate_angle(a, b, c):
    print(a)
    print(type(a))
    a = np.array(a)  # First
    print('numpy')
    print(a)
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_position(shoulder, elbow, counter=0):
    if shoulder > elbow:
        counter += 1
        stage = "up"
        return stage
    elif shoulder < elbow:
        stage = "down"
        return stage


def write_image_message(stage, position):
    if stage == 'up' and position == 'right':
        cv2.rectangle(image, (0, 0), (225, 93), (26, 21, 223), -1)
    elif stage == 'down' and position == 'right':
        cv2.rectangle(image, (0, 0), (225, 93), (77, 100, 17), -1)
    elif stage == 'up' and position == 'left':
        cv2.rectangle(image, (0, 190), (225, 96), (56, 21, 223), -1)
    elif stage == 'down' and position == 'left':
        cv2.rectangle(image, (0, 190), (225, 96), (100, 100, 17), -1)

    # Rep data
    if position == 'right':
        cv2.putText(image, '{} Hand'.format(position.capitalize()), (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'Status : {}'.format(stage), (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Raise Count : {}".format(stage),
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    else:
        cv2.putText(image, '{} Hand'.format(position.capitalize()), (15, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'Status : {}'.format(stage), (15, 147),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Raise Count : {}".format(stage),
                    (15, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Curl counter variables
# counter = 0
# stage = None

coordinates_file_list = []

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(lshoulder, lelbow, lwrist)
            # print(angle)
            rangle = calculate_angle(rshoulder, relbow, rwrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(lelbow, [1280, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            # Visualize angle
            cv2.putText(image, str(rangle),
                        tuple(np.multiply(relbow, [1280, 720]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            """if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                print(counter)"""
            # Right Shoulder position
            right_stage = get_position(rshoulder[1], relbow[1])
            # Left Shoulder position
            left_stage = get_position(lshoulder[1], lelbow[1])
        except:
            pass

        # Setup status box
        write_image_message(right_stage, 'right')
        write_image_message(left_stage, 'left')

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Shoulders Position', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

