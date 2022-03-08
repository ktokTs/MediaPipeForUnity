from asyncio.windows_events import NULL
from email import message
import socket
import random
import time
import os
import json
import mediapipe as mp
import cv2
import logging
import datetime
from connectunity import connectunity
from mp_hand import mp_hand
from mp_face import mp_face

HOST = "127.0.0.1"
MAINPORT = 50007

is_connectunity = True

landmark_line_ids = []

fh = logging.FileHandler('log.log')
logger = logging.getLogger('Logging')
logger.addHandler(fh)

def init_mp(num):
    global landmark_line_ids
    if num == 0:
        mediapipe = mp_hand()
    if num == 1:
        mediapipe = mp_face()
    landmark_line_ids = mediapipe.landmark_line_ids
    print(landmark_line_ids)

    return mediapipe

def GetLandmarks(mediapipe, cap):
    global landmark_line_ids

    if cap.isOpened():
        # カメラから画像取得
        success, img = cap.read()
        if not success:
            sendlog(30, "cap.read() fail")
            return NULL
        img = cv2.flip(img, 1)          # 画像を左右反転
        img_h, img_w, _ = img.shape     # サイズ取得

        Res = NULL
        # 検出処理の実行
        mediapipe.Process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 検出した手の数分繰り返し
        if mediapipe.AllDetections():
            for h_id, hand_landmarks in enumerate(mediapipe.GetLandmarks(0)):
                Res = make_landmarks_json(hand_landmarks, img_h, img_w)
            print_landmarks(mediapipe, img, img_w, img_h)

        # 画像の表示
        cv2.imshow("MediaPipe Hands", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            return NULL
    else:
        sendlog(30, "cap.isOpened() is false")
    return Res

def print_landmarks(mediapipe, img, img_w, img_h):
    for h_id, hand_landmarks in enumerate(mediapipe.GetLandmarks(1)):
    # landmarkの繋がりをlineで表示
        for line_id in landmark_line_ids:
            # 1点目座標取得
            lm = hand_landmarks.landmark[line_id[0]]
            lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
            # 2点目座標取得
            lm = hand_landmarks.landmark[line_id[1]]
            lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
            # line描画
            cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)

        # landmarkをcircleで表示
        z_list = [lm.z for lm in hand_landmarks.landmark]
        z_min = min(z_list)
        z_max = max(z_list)
        for id, lm in enumerate(hand_landmarks.landmark):
            lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
            lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
            if id in mediapipe.GetPutCircle():
                cv2.circle(img, lm_pos, 3, (100, lm_z, lm_z), -1)
            else:
                continue
                cv2.circle(img, lm_pos, 3, (255, lm_z, lm_z), -1)
        putdetectinfo(mediapipe, h_id, img_w, img_h, img)

def putdetectinfo(mediapipe, h_id, img_w, img_h, img):
    # 検出情報をテキスト出力
    # - テキスト情報を作成
    if (mediapipe.IsInfo == False):
        return
    hand_texts = mediapipe.InfoTexts(h_id)
    lm = mediapipe.InfoDisplayPoint(h_id)
    lm_x = int(lm.x * img_w) - 50
    lm_y = int(lm.y * img_h) - 10
    lm_c = (64, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # - テキスト出力
    for cnt, text in enumerate(hand_texts):
        cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)

def make_landmarks_json(hand_landmarks, img_h, img_w):
    res = []
    index = 0
    for lm in (hand_landmarks.landmark):
        data_p = {}
        data = {}

        data_p['x'] = lm.x
        data_p['y'] = lm.y
        data_p['z'] = lm.z
        data['Index'] = index
        data['Point'] = data_p
        # print(index, data_p['x'], data_p['y'], data_p['z'])
        res.append(data)
        index += 1
    return res

def main():
    unityconnecter = connectunity(HOST, MAINPORT, is_connectunity)
    unityconnecter.ConnectUnity()

    mediapipe = init_mp(1)
    # unityconnecter.Receive(200)

    cap = cv2.VideoCapture(0)   # カメラのID指定
    try:
        while True:
            data = GetLandmarks(mediapipe, cap)
            json_data = json.dumps(data)
            #print(json_data)
            unityconnecter.Send(json_data.encode('utf-8'))
            time.sleep(0.2)
    except ConnectionAbortedError:
        print("Connection aborte")
    finally:
        cap.release()

def sendlog(num, string):
    now =  str(datetime.datetime.now(datetime.timezone.utc))
    logger.log(num, now + "> " + string)

if __name__ == "__main__":
    main()