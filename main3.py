import cv2
import PoseModule as pm
import time
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from gpiozero import Buzzer
from time import sleep
from libcamera import controls


from PoseModule import PoseDetector
buzzer = Buzzer(18)
in1 = 23
in2 = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)




def move_up():
    # Add code to move the linear actuator up
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    print("Moving up")

def move_down():
    # Add code to move the linear actuator down
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    print("Moving down")

def stop_motor():
    # Add code to stop the linear actuator
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    print("Stopping motor")
    
cv2.startWindowThread() #optional
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, raw=picam2.sensor_modes[2]))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

with open('RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

detector = pm.PoseDetector()
start_time = time.time()
height_data = []
elapsed_time = 0
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
current_pose_data_name = None
pose_names = {ord('1'): 'ArcheryA', ord('2'): 'ArcheryB', ord('3'): 'ArcheryC'}
try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            img = picam2.capture_array()
            img = cv2.resize(img, (800, 600))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = detector.findPose(img, False)
            lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
            cv2.rectangle(img, (0, 0), (800, 200), (225, 0, 0), 2)  # reff box
            cv2.rectangle(img, (580, 500), (790, 590), (245, 117, 16), -1)
            cv2.rectangle(img, (10, 5), (410, 130), (245, 117, 16), -1)
            cv2.putText(img, "INSTRUCTION:-", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46), 2)

            results = holistic.process(img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key in pose_names:
                current_pose_data_name = pose_names[key]
                print(f"Changed mapping pose to: {current_pose_data_name}")

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Concate rows
                row = pose_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                # print(body_language_class, body_language_prob)
                cv2.rectangle(img, (580, 20), (790, 100), (245, 117, 16), -1)

                # Display Class
                #cv2.putText(img, 'CLASS'
                #            , (675, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                #cv2.putText(img, body_language_class.split(' ')[0]
                #            , (675, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(img, 'ACCURACY'
                            , (595, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (590, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass
            if lmList and current_pose_data_name:
                try:
                    print(f"Calling mapping function for {current_pose_data_name}")
                    getattr(detector, f'mapping_{current_pose_data_name}')(img)
                    print("Mapping function called successfully")
                except Exception as e:
                    print(f"Error in mapping function: {e}")
                
                # Get the center of the bounding box around the body
                center = bboxInfo["center"]
                tinggi = bboxInfo["Height"]
                line = bboxInfo["Reff"]
                if line > 210:
                    move_down()

                if line < 190:
                    move_up()

                if 191 < line < 209:
                    stop_motor()

                # Calculate the angle between landmarks
                angle0, img = detector.findAngle(lmList[16][0:2], lmList[14][0:2], lmList[12][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)

                angle1, img = detector.findAngle(lmList[11][0:2], lmList[13][0:2], lmList[15][0:2],
                                                 img=img,color=(0, 0, 255), scale=10)

                angle2, img = detector.findAngle(lmList[12][0:2], lmList[11][0:2],  lmList[13][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)

                angle3, img = detector.findAngle(lmList[14][0:2], lmList[12][0:2], lmList[11][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)
                angle4, img = detector.findAngle(lmList[11][0:2], lmList[23][0:2], lmList[25][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)

                angle5, img = detector.findAngle(lmList[12][0:2], lmList[24][0:2], lmList[26][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)

                angle6, img = detector.findAngle(lmList[24][0:2], lmList[26][0:2], lmList[28][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)
                angle7, img = detector.findAngle(lmList[23][0:2], lmList[25][0:2], lmList[27][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)
                angle8, img = detector.findAngle(lmList[14][0:2], lmList[20][0:2], lmList[4][0:2],
                                                 img=img, color=(0, 0, 255), scale=10)

                # Check if the angle is close to 50 degrees with an offset of 5
                leftKnee = detector.angleCheck(myAngle=angle7, targetAngle=180, offset=10)
                if leftKnee == False:
                    cv2.putText(img, "Please keep your left knee straight", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "Your Left Knee Are Ready", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                rightKnee = detector.angleCheck(myAngle=angle6, targetAngle=180, offset=10)
                if rightKnee == False:
                    cv2.putText(img, "Please keep your left knee straight", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "Your Right Knee Are Ready", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                leftElbow = detector.angleCheck(myAngle=angle1, targetAngle=180, offset=10)
                if leftElbow == False:
                    cv2.putText(img, "Please straiten your bow arms", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "Your Bow arms Are Ready", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                leftShoulder = detector.angleCheck(myAngle=angle2, targetAngle=180, offset=10)
                if leftShoulder == False:
                    cv2.putText(img, "Please align your shoulders with your bow arms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "Your Shoulders Are Ready", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                rightShoulder = detector.angleCheck(myAngle=angle3, targetAngle=150, offset=10)
                if rightShoulder == False:
                    cv2.putText(img, "Please keep your forearm horizontally", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "your forearm is ready", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                    
                anchorpoint = detector.angleCheck(myAngle=angle8, targetAngle=90, offset=20)
                if anchorpoint == False:
                    cv2.putText(img, "Please keep anchorpoint", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 46, 46),
                                2)
                else:
                    cv2.putText(img, "your ancher point are ready", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                if leftKnee and rightKnee and leftShoulder and rightShoulder and leftElbow and anchorpoint == True:
                    cv2.putText(img, "GOOD POSE", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    buzzer.on()
                    sleep(0.5)
                    buzzer.off()
                else:
                    cv2.putText(img, "BAD POSE", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    buzzer.off()
                    sleep(1)
                height_data.append(tinggi)
                # Calculate the elapsed time
                elapsed_time = time.time() - start_time
                # Check if 10 seconds have passed
                if height_data:
                    average = detector.calcAvgHeight(height_data)
                    # print("Average Height:", average)
                    cv2.putText(img, "Height: {} cm".format(round(average, 2)), (590, 520),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            if elapsed_time >= 10:
                # Reset the timer and height data list
                start_time = time.time()
                height_data = []

            cv2.putText(img, f"Current Pose: {current_pose_data_name}", (590, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow("Imager", img)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
    print("done")