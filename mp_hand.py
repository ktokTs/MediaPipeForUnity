from http import client
import mediapipe as mp

class mp_hand:
    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=1,                # 最大検出数
            min_detection_confidence=0.7,   # 検出信頼度
            min_tracking_confidence=0.7     # 追跡信頼度
        )
        self.IsInfo = True

        # landmarkの繋がり表示用
        self.landmark_line_ids = [ 
            (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
            (1, 2), (2, 3), (3, 4),         # 親指
            (5, 6), (6, 7), (7, 8),         # 人差し指
            (9, 10), (10, 11), (11, 12),    # 中指
            (13, 14), (14, 15), (15, 16),   # 薬指
            (17, 18), (18, 19), (19, 20),   # 小指
        ]

        self.putcircle = range(20)

    def Process(self, img):
        self.results = self.hands.process(img)

    def AllDetections(self):
        return self.results.multi_hand_world_landmarks

    def GetLandmarks(self, num):
        if num == 0:
            return self.results.multi_hand_world_landmarks
        if num == 1:
            return self.results.multi_hand_landmarks
        return

    def GetPutCircle(self):
        return self.putcircle

    def InfoTexts(self, h_id):
        hand_texts = []
        for c_id, hand_class in enumerate(self.results.multi_handedness[h_id].classification):
            hand_texts.append("#%d-%d" % (h_id, c_id)) 
            hand_texts.append("- Index:%d" % (hand_class.index))
            hand_texts.append("- Label:%s" % (hand_class.label))
            hand_texts.append("- Score:%3.2f" % (hand_class.score * 100))
        return hand_texts

    def InfoDisplayPoint(self, h_id):
        return self.results.multi_hand_landmarks[h_id].landmark[0]

