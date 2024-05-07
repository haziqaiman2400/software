import math
import cv2
import mediapipe as mp
import time

class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, staticMode=False,
                 modelComplexity=1,
                 smoothLandmarks=True,
                 enableSegmentation=False,
                 smoothSegmentation=True,
                 detectionCon=0.5,
                 trackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smoothLandmarks,
                                     enable_segmentation=self.enableSegmentation,
                                     smooth_segmentation=self.smoothSegmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][0] - self.lmList[11][0]) // 2

            if bboxWithHands:
                x1 = self.lmList[16][0] - ad
                x2 = self.lmList[15][0] + ad
            else:
                x1 = self.lmList[12][0] - ad
                x2 = self.lmList[11][0] + ad

            y2 = self.lmList[29][1] + ad
            y1 = self.lmList[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2
            # Calculate height
            PPI = 96
            pixel_width = x2 - x1
            pixel_to_cm = pixel_width / PPI
            height = ((y2 - y1) * pixel_to_cm / 2.54)*0.65

            # create reference point
            p_x = x1, y1
            p_y = x2, y1
            line = y1
            # print("Line:" + str(line))
            # print(format(round(height, 2)))
            self.bboxInfo = {"bbox": bbox, "center": (cx, cy), "Height": height, "Reff": line}
            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.line(img, p_x, p_y, (255, 0, 0), 2)
                # cv2.putText(img, "Height: {} cm".format(round(height, 2)), (bbox[0], bbox[1] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return self.lmList, self.bboxInfo

    def mapping_ArcheryA(self, img):

        color = (161, 20, 151)
        line_color = (0, 255, 0)
        p1 = (600, 260)
        p2 = (500, 260)
        p3 = (400, 260)
        p4 = (300, 260)
        p5 = (230, 230)
        p6 = (330, 230)
        p7 = (305, 400)
        p8 = (300, 470)
        p9 = (300, 570)
        p10 = (390, 400)
        p11 = (400, 470)
        p12 = (400, 570)

        cv2.line(img, p1, p2, line_color, 2)
        cv2.line(img, p2, p3, line_color, 2)
        cv2.line(img, p3, p4, line_color, 2)
        cv2.line(img, p4, p5, line_color, 2)
        cv2.line(img, p5, p6, line_color, 2)
        cv2.line(img, p4, p7, line_color, 2)
        cv2.line(img, p7, p8, line_color, 2)
        cv2.line(img, p8, p9, line_color, 2)
        cv2.line(img, p3, p10, line_color, 2)
        cv2.line(img, p10, p11, line_color, 2)
        cv2.line(img, p11, p12, line_color, 2)

        cv2.circle(img, p1, 5, color, cv2.FILLED)
        cv2.circle(img, p2, 5, color, cv2.FILLED)
        cv2.circle(img, p3, 5, color, cv2.FILLED)
        cv2.circle(img, p5, 5, color, cv2.FILLED)
        cv2.circle(img, p4, 5, color, cv2.FILLED)
        cv2.circle(img, p6, 5, color, cv2.FILLED)
        cv2.circle(img, p7, 5, color, cv2.FILLED)
        cv2.circle(img, p8, 5, color, cv2.FILLED)
        cv2.circle(img, p9, 5, color, cv2.FILLED)
        cv2.circle(img, p10, 5, color, cv2.FILLED)
        cv2.circle(img, p11, 5, color, cv2.FILLED)
        cv2.circle(img, p12, 5, color, cv2.FILLED)

        return img
    def mapping_ArcheryB(self, img):

        color = (161, 20, 151)
        line_color = (22, 220, 0)
        p1 = (580, 260)
        p2 = (510, 260)
        p3 = (460, 260)
        p4 = (400, 260)
        p5 = (370, 240)
        p6 = (430, 250)
        p7 = (406, 400)
        p8 = (400, 470)
        p9 = (400, 570)
        p10 = (450, 400)
        p11 = (460, 470)
        p12 = (460, 570)

        cv2.line(img, p1, p2, line_color, 2)
        cv2.line(img, p2, p3, line_color, 2)
        cv2.line(img, p3, p4, line_color, 2)
        cv2.line(img, p4, p5, line_color, 2)
        cv2.line(img, p5, p6, line_color, 2)
        cv2.line(img, p4, p7, line_color, 2)
        cv2.line(img, p7, p8, line_color, 2)
        cv2.line(img, p8, p9, line_color, 2)
        cv2.line(img, p3, p10, line_color, 2)
        cv2.line(img, p10, p11, line_color, 2)
        cv2.line(img, p11, p12, line_color, 2)

        cv2.circle(img, p1, 5, color, cv2.FILLED)
        cv2.circle(img, p2, 5, color, cv2.FILLED)
        cv2.circle(img, p3, 5, color, cv2.FILLED)
        cv2.circle(img, p5, 5, color, cv2.FILLED)
        cv2.circle(img, p4, 5, color, cv2.FILLED)
        cv2.circle(img, p6, 5, color, cv2.FILLED)
        cv2.circle(img, p7, 5, color, cv2.FILLED)
        cv2.circle(img, p8, 5, color, cv2.FILLED)
        cv2.circle(img, p9, 5, color, cv2.FILLED)
        cv2.circle(img, p10, 5, color, cv2.FILLED)
        cv2.circle(img, p11, 5, color, cv2.FILLED)
        cv2.circle(img, p12, 5, color, cv2.FILLED)

        return img
    def mapping_ArcheryC(self, img):

        color = (161, 20, 151)
        line_color = (47, 132, 11)
        p1 = (600, 200)
        p2 = (500, 200)
        p3 = (400, 200)
        p4 = (300, 200)
        p5 = (230, 170)
        p6 = (330, 170)
        p7 = (300, 370)
        p8 = (300, 470)
        p9 = (300, 570)
        p10 = (400, 370)
        p11 = (400, 470)
        p12 = (400, 570)

        cv2.line(img, p1, p2, line_color, 2)
        cv2.line(img, p2, p3, line_color, 2)
        cv2.line(img, p3, p4, line_color, 2)
        cv2.line(img, p4, p5, line_color, 2)
        cv2.line(img, p5, p6, line_color, 2)
        cv2.line(img, p4, p7, line_color, 2)
        cv2.line(img, p7, p8, line_color, 2)
        cv2.line(img, p8, p9, line_color, 2)
        cv2.line(img, p3, p10, line_color, 2)
        cv2.line(img, p10, p11, line_color, 2)
        cv2.line(img, p11, p12, line_color, 2)

        cv2.circle(img, p1, 5, color, cv2.FILLED)
        cv2.circle(img, p2, 5, color, cv2.FILLED)
        cv2.circle(img, p3, 5, color, cv2.FILLED)
        cv2.circle(img, p5, 5, color, cv2.FILLED)
        cv2.circle(img, p4, 5, color, cv2.FILLED)
        cv2.circle(img, p6, 5, color, cv2.FILLED)
        cv2.circle(img, p7, 5, color, cv2.FILLED)
        cv2.circle(img, p8, 5, color, cv2.FILLED)
        cv2.circle(img, p9, 5, color, cv2.FILLED)
        cv2.circle(img, p10, 5, color, cv2.FILLED)
        cv2.circle(img, p11, 5, color, cv2.FILLED)
        cv2.circle(img, p12, 5, color, cv2.FILLED)

        return img
    def calcAvgHeight(self, height_data):
        """
        Calculate the average height of the first 10 height data.
        :param height_data: List of height data.
        """
        if isinstance(height_data, list):
            avg_height = sum(height_data[:10]) / 10
            return avg_height
        else:
            # Handle the case where height_data is a single float value
            return height_data

    def findAngle(self,p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        """
        Finds angle between three points.

        :param p1: Point1 - (x1,y1)
        :param p2: Point2 - (x2,y2)
        :param p3: Point3 - (x3,y3)
        :param img: Image to draw output on. If no image input output img is None
        :return:
        """

        # Get the landmarks
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1,scale//5))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x1, y1), scale+5, color, max(1,scale//5))
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale+5, color, max(1,scale//5))
            cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), scale+5, color, max(1,scale//5))
            # cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
            #             cv2.FONT_HERSHEY_PLAIN, 2, color, max(1,scale//5))
        return angle, img

    def angleCheck(self, myAngle, targetAngle, offset=20):
        return targetAngle - offset < myAngle < targetAngle + offset


def main():
    # Initialize the webcam and set it to the third camera (index 2)
    cap = cv2.VideoCapture(0)

    # Initialize the PoseDetector class with the given parameters
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    # Loop to continuously get frames from the webcam
    start_time = time.time()
    height_data = []
    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        # Find the human pose in the frame
        img = detector.findPose(img, draw=False)

        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
        line = bboxInfo["Reff"]
        print(line)
        cv2.rectangle(img, (0, 0), (800, 150), (225, 0, 0), 2)
        if line > 160:
            print("move down")

        if line < 141:
            print("move up")

        if 160 > line > 139:
            print("stop")

        # Check if any body landmarks are detected
        if lmList:
            # Get the center of the bounding box around the body
            center = bboxInfo["center"]
            tinggi = bboxInfo["Height"]
            # print("Tinggi:" + format(round(tinggi, 2)))
            # Draw a circle at the center of the bounding box
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # Calculate the angle between landmarks 11, 13, and 15 and draw it on the image
            angle, img = detector.findAngle(lmList[11][0:2],
                                            lmList[13][0:2],
                                            lmList[15][0:2],
                                            img=img,
                                            color=(0, 0, 255),
                                            scale=10)

            # Check if the angle is close to 50 degrees with an offset of 10
            isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                                 targetAngle=50,
                                                 offset=10)

            # Print the result of the angle check
            #print(isCloseAngle50)
            height_data.append(tinggi)

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time
            # Check if 10 seconds have passed
            if height_data:
                average = detector.calcAvgHeight(height_data)
                #print("Average Height:", average)
                cv2.putText(img, "Height {} cm".format(round(average, 2)), (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        if elapsed_time >= 10:
            # Reset the timer and height data list
            start_time = time.time()
            height_data = []

        # Wait for 1 millisecond between each frame
        cv2.imshow("Result", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
