import cv2
import numpy as np
import PoseDetector
import time
class PoseModel:
    def __init__(self):
        pass
    def posemodel(self):
        L_count = 0
        R_count = 0
        pTime = 0
        #When the hand is moving up
        dir = 0
        # cap = cv2.VideoCapture("/Users/aravinthan/PycharmProjects/OpenCV_Tutorial/597/video2.mp4")
        cap = cv2.VideoCapture(0)
        detector = PoseDetector.poseDetectorkit()
        while cap.isOpened():
            print("Video Capturing Preprocessing : ")
            success,img = cap.read()
            print("Video Capture Inferring : ")
            #Finding the Pose
            img = detector.findPose(img)
            print('test')
            #Finding the Landmarks
            landmarklist = detector.findlandmarks(img)
            if len(landmarklist)!=0:
                # Left Arm Points
                Left_H_angle = detector.findAngle(img, 11, 13, 15)
                print(Left_H_angle)
                Right_H_angle = detector.findAngle(img, 12, 14, 16)
                print(Right_H_angle)
                L_per = np.interp(Left_H_angle,(0, 210),(0,100))
                R_per = np.interp(Right_H_angle, (0, 210), (0, 100))
                print(Left_H_angle, L_per)
                # Check for the Right_dumbell curves
                if (L_per) == 0.5:
                    if dir == 0:
                        L_count += 0.5
                        dir = 1
                if (L_per) ==0.5:
                    if dir ==1:
                        L_count+=0.5
                        dir = 0
                print(L_count,R_count)
                # # Curl counter logic
                # if Left_angle > 160:
                #     stage = "down"
                # if Left_angle < 30 and stage == 'down':
                #     stage = "up"
                #     L_count += 1
                #     print(L_count)

            # cv2.circle(img,(landmarklist[12][1], landmarklist[12][2]),15,(0,0,255),cv2.FILLED)
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
            cv2.imshow('Image',img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

PoseModel().posemodel()