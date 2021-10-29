import cv2
import numpy as np
import PoseDetector
import time
class PoseModel:
    def __init__(self):
        self.sleep = 20

    def posemodel(self):

        pTime = 0
        # cap = cv2.VideoCapture("/Users/aravinthan/PycharmProjects/OpenCV_Tutorial/597/video2.mp4")
        cap = cv2.VideoCapture(0)
        detector = PoseDetector.poseDetectorkit()
        poseCorrect=False
        while cap.isOpened():
            self.sleep -= 1
            # print("Video Capturing Preprocessing : ")
            success,img = cap.read()
            # print("Video Capture Inferring : ")
            #Finding the Pose
            img = detector.findPose(img)
            #Finding the Landmarks
            landmarklist = detector.findlandmarks(img)
            if len(landmarklist)!=0:
                # Left Arm Points
                Left_H_angle = detector.findAngle(img, 11, 13, 15)
                Right_H_angle = detector.findAngle(img, 12, 14, 16)
                # L_per = np.interp(Left_H_angle,(0, 210),(0,100))
                # R_per = np.interp(Right_H_angle, (0, 210), (0, 100))
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
            cv2.imshow('Image',img)
            if self.sleep==0:
                # break
                pass
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
