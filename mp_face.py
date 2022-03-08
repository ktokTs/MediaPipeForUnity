from asyncio.windows_events import NULL
from http import client
import os
import socket
import mediapipe as mp
import cv2

class mp_face:
    def __init__(self):
        mp_solution = mp.solutions.face_mesh
        self.solution = mp_solution.FaceMesh(
            refine_landmarks=True
        )
        self.IsInfo = False

        # landmarkの繋がり表示用
        connections = mp.solutions.face_mesh_connections
        self.landmark_line_ids = list(connections.FACEMESH_RIGHT_IRIS) # 右目虹彩
        self.landmark_line_ids += (list(connections.FACEMESH_LEFT_IRIS)) # 右目虹彩
        self.landmark_line_ids += (list(connections.FACEMESH_LEFT_EYE)) # 右目周り
        self.putcircle = (
            473, # 左目の中央
            263, # 左目じり
            362  # 左目頭
            )

    def Process(self, img):
        self.results = self.solution.process(img)

    def AllDetections(self):
        return self.results.multi_face_landmarks

    def GetLandmarks(self, num):
        if num == 0:
            return self.results.multi_face_landmarks
        if num == 1:
            return self.results.multi_face_landmarks
        return

    def GetPutCircle(self):
        return self.putcircle

    def InfoTexts(self, h_id):
        return

    def InfoDisplayPoint(self, h_id):
        return

