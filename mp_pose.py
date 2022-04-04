from http import client
import mediapipe as mp

class mp_pose:
    def __init__(self):
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.2,   # 検出信頼度
            min_tracking_confidence=0.2     # 追跡信頼度
        )
        self.IsInfo = False

        # landmarkの繋がり表示用
        self.landmark_line_ids = [ 
            (12, 11), (11 ,23), (23 ,24), (24, 12),
            (12, 14), (14, 16), (16, 22), (16, 20), (16, 18), (18, 20),
            (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19), 
            (24, 26), (26, 28), (28, 32), (28, 30), (32, 30), 
            (23, 25), (25, 27), (27, 31), (27, 29), (31, 29)
        ]

        self.putcircle = range(11, 32, 1)

    def Process(self, img):
        self.results = self.pose.process(img)

    def AllDetections(self):
        return self.results.pose_world_landmarks

    def GetLandmarks(self, num):
        if num == 0:
            return [self.results.pose_world_landmarks]
        if num == 1:
            return [self.results.pose_landmarks]
        return

    def GetPutCircle(self):
        return self.putcircle

    def InfoTexts(self, h_id):
        return 

    def InfoDisplayPoint(self, h_id):
        return 

