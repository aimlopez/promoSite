import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detectionCon=0, trackCon=0):  # constructor
        self.results = None
        self.lmlist = None
        self.mode = mode
        self.max_hands = max_hands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mp_hands = mp.solutions.hands  # initializing hands module for the instance
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detectionCon,
                                         self.trackCon)  # object for Hands for a particular instance
        self.mp_draw = mp.solutions.drawing_utils  # object for Drawing
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting to RGB bcoz hand recognition works only on RGB image
        self.results = self.hands.process(imgRGB)  # processing the RGB image
        if self.results.multi_hand_landmarks:  # gives x,y,z of every landmark or if no hand than NONE
            for handLms in self.results.multi_hand_landmarks:  # each hand landmarks in results
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []
        if self.results.multi_hand_landmarks:  # gives x,y,z of every landmark
            my_hand = self.results.multi_hand_landmarks[handNo]  # Gives result for particular hand
            for id, lm in enumerate(my_hand.landmark):  # gives id and lm(x,y,z)
                h, w, c = img.shape  # getting h,w for converting decimals x,y into pixels
                cx, cy = int(lm.x * w), int(lm.y * h)  # pixels coordinates for landmarks
                # print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                # if draw:
                #    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmlist, bbox

    def fingers_up(self):  # checking which finger is open
        fingers = []  # storing final result
        # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign
        # checking y position of 4 is in right to y position of 8
        if self.lmlist[self.tipIds[0]][2] < self.lmlist[self.tipIds[1] - 3][2]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):  # checking tip point is below tippoint-2 (only in Y direction)
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 3][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # totalFingers = fingers.count(1)
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):  # finding distance between two points p1 & p2
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]  # getting x,y of p1
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]  # getting x,y of p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # getting centre point

        if draw:  # drawing line and circles on the points
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # PTime = 0  # previous time
    # CTime = 0  # current time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector()

    while True:
        success, img = cap.read()  # T or F,frame
        img = detector.find_hands(img)
        lmlist, bbox = detector.find_position(img)

        if len(lmlist) != 0:
            #print(lmlist[4])
            hands_up = detector.fingers_up()
            print(hands_up)
            if any(i == 1 for i in hands_up):
                x, y, w, h = 0, 0, 640, 120
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.putText(img, 'Hand Up', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, 'WELL DONE', (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                x, y, w, h = 0, 0, 640, 120
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                cv2.putText(img, 'Hand down', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, 'Return to Basic Skills', (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1, cv2.LINE_AA)

        else:
            cv2.putText(img, "Press 'q' over the screen to exit", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Hur lifter du föremålet', img)  # showing img not imgRGB
        cv2.setWindowProperty('Hur lifter du föremålet', cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
