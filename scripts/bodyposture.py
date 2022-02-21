import cv2
import mediapipe as mp
import numpy as np


class PostureDetector:
    def __init__(self, detection_conf=0, track_conf=0):
        self.results = None
        self.landmark_list = None
        self.stage = None
        self.drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.pose = self.mp_pose.Pose(self.detection_conf, self.track_conf)

    def find_pose(self, image, draw=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.results.pose_landmarks:
            self.drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                        self.mp_drawing_styles.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                           circle_radius=2),
                                        self.mp_drawing_styles.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                           circle_radius=2))
        return image

    def extract_landmarks(self):
        self.landmark_list = {}

        try:
            landmarks = self.results.pose_landmarks.landmark

            # Get coordinates left side
            self.landmark_list['lshoulder'] = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                               landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            self.landmark_list['lelbow'] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            self.landmark_list['lwrist'] = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            self.landmark_list['lhip'] = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # Get coordinates right side
            self.landmark_list['rshoulder'] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            self.landmark_list['relbow'] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            self.landmark_list['rwrist'] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            self.landmark_list['rhip'] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            return self.landmark_list
        except:
            pass

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return round(angle, 2)

    def get_position(self, shoulder, elbow, counter=0):
        if shoulder > elbow:
            counter += 1
            stage = "up"
            return stage
        elif shoulder < elbow:
            stage = "down"
            return stage

    # TODO: change this function to object oriented programming
    def get_angle(self, angle, counter=0):
        if angle > 90.0:
            counter += 1
            stage = "up"
            return stage
        elif angle < 90.0:
            stage = "down"
            return stage

    # TODO: change this function to object oriented programming
    def write_image_message(self, image, stage, position, angle):
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
            cv2.putText(image, "Angle : {}".format(angle),
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            cv2.putText(image, '{} Hand'.format(position.capitalize()), (15, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Status : {}'.format(stage), (15, 147),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Angle : {}".format(angle),
                        (15, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = PostureDetector()

    while True:
        ret, image = cap.read()
        image = detector.find_pose(image)
        lmlist = detector.extract_landmarks()

        # Calculate angle
        left_angle = detector.calculate_angle(lmlist['lhip'], lmlist['lshoulder'], lmlist['lelbow'])
        right_angle = detector.calculate_angle(lmlist['rhip'], lmlist['rshoulder'], lmlist['relbow'])

        # Visualize left angle
        cv2.putText(image, str(left_angle),
                    tuple(np.multiply(lmlist['lshoulder'], [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        # Visualize right angle
        cv2.putText(image, str(right_angle),
                    tuple(np.multiply(lmlist['rshoulder'], [1280, 720]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Right elbow position in relation with shoulder position
        right_stage = detector.get_position(lmlist['rshoulder'][1], lmlist['relbow'][1])

        # Left elbow position in relation with shoulder position
        left_stage = detector.get_position(lmlist['lshoulder'][1], lmlist['lelbow'][1])

        # Setup status box
        detector.write_image_message(image, right_stage, 'right', right_angle)
        detector.write_image_message(image, left_stage, 'left', left_angle)

        cv2.imshow('Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
