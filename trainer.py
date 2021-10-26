import cv2
import PoseModule
import time
cap = cv2.VideoCapture("/Users/aravinthan/PycharmProjects/OpenCV_Tutorial/597/video1.mp4")
# cap = cv2.VideoCapture(0)
# app=Flask(__name__)
pTime = 0
detector = PoseModule.poseDetector()
while True:
    print("Video Capturing Preprocessing : ")
    success,img = cap.read()
    print("Video Capture Inferring : ")
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList)!=0:
        detector.findAngle(img,12,14,16)
        detector.findAngle(img, 11, 13, 15)

    print(lmList)
    cv2.circle(img,(lmList[12][1], lmList[12][2]),15,(0,0,255),cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    print("Result shown ")
    cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)
# def myModel():
#     while True:
#         print("Video Capturing Preprocessing : ")
#         success,img = cap.read()
#         print("Video Capture Inferring : ")
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img)
#         if len(lmList)!=0:
#             detector.findAngle(img,12,14,16)
#             detector.findAngle(img, 11, 13, 15)
#
#         print(lmList)
#         cv2.circle(img,(lmList[12][1], lmList[12][2]),15,(0,0,255),cv2.FILLED)
#         cTime = time.time()
#         fps = 1/(cTime - pTime)
#         pTime = cTime
#         print("Result shown ")
#         cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
#         cv2.imshow('Image',img)
#         if cv2.waitKey(1) & 0XFF == ord('q'):
#             break
#     cap.release()


# def respond_success():
#     return "success"
# @app.route("/")
# def base():
#     t1 = threading.Thread(target=myModel, args=(10,))
#     t2 = threading.Thread(target=respond_success, args=(10,))
#     t2.start()
#     t1.start()
#     print("Executed")
#
# app.run(host="0.0.0.0",port=81)