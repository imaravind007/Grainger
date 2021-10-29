import cv2
import mediapipe as mp
import numpy as np
import math

class poseDetectorkit():
    def __init__(self,static_mode=False,upperbody = False,smooth = True,
                 detectionConfidence = 0.5, trackConfidence=0.5):
        self.static_mode = static_mode
        self.upperbody = upperbody
        self.smooth=smooth
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()


    #Creatign a method to find the pose:

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    # Creatign a method to find the landmarks:

    def findlandmarks(self, img, draw=True):
        self.landmarkList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img,(cx, cy), 5 ,(0,0,255),cv2.FILLED)
        return self.landmarkList

    def findAngle(self, img,p1,p2,p3, draw = True):
        #Landmarks
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        x3, y3 = self.landmarkList[p3][1:]

        #Angle Calculation
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                          math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle+=360

        if draw:
            #Line taken at that particular point
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.line(img,(x2, y2),(x3, y3),(0, 255, 0),3)
            #Circle for the hand (small and the big circle)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1, y1), 30, (0, 0, 255),2)
            cv2.circle(img,(x2, y2), 10,(0, 0, 255),cv2.FILLED)
            cv2.circle(img,(x2, y2), 30, (0, 0, 255),2)
            cv2.circle(img,(x3, y3), 10,(0, 0, 255),cv2.FILLED)
            cv2.circle(img,(x3, y3), 30, (0, 0, 255),2)
            #Writting the angle in the exact place of the image
            cv2.putText(img, str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return angle