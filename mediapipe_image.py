#python送信側

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

HOST = "127.0.0.1"
MAINPORT = 50007
SEND_PID_PORT = 50006

connectunity = False

landmark_line_ids = []

fh = logging.FileHandler('test.log')
logger = logging.getLogger('LoggingTest')
logger.addHandler(fh)

def GetSimpleData():
    a = random.randrange(3)
    data = {}
    data["Count"] = 0
    data["Num"] = a
    result = str(a)
    print(result)
    return data

def init_mp():
    global landmark_line_ids

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,                # 最大検出数
        min_detection_confidence=0.7,   # 検出信頼度
        min_tracking_confidence=0.7     # 追跡信頼度
    )

    # landmarkの繋がり表示用
    landmark_line_ids = [ 
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
        (1, 2), (2, 3), (3, 4),         # 親指
        (5, 6), (6, 7), (7, 8),         # 人差し指
        (9, 10), (10, 11), (11, 12),    # 中指
        (13, 14), (14, 15), (15, 16),   # 薬指
        (17, 18), (18, 19), (19, 20),   # 小指
    ]

    cap = cv2.VideoCapture(0)   # カメラのID指定

    return hands, cap

def GetHands(hands, cap):
    global landmark_line_ids

    if cap.isOpened():
        # カメラから画像取得
        imgname = input()
        img = cv2.imread(imgname)
        img = cv2.flip(img, 1)          # 画像を左右反転
        img_h, img_w, _ = img.shape     # サイズ取得

        # 検出処理の実行
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            # 検出した手の数分繰り返し
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                res = make_hand_landmarks_json(hand_landmarks)
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
                for lm in hand_landmarks.landmark:
                    lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                    lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                    cv2.circle(img, lm_pos, 3, (255, lm_z, lm_z), -1)

                # 検出情報をテキスト出力
                # - テキスト情報を作成
                hand_texts = []
                for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                    hand_texts.append("#%d-%d" % (h_id, c_id)) 
                    hand_texts.append("- Index:%d" % (hand_class.index))
                    hand_texts.append("- Label:%s" % (hand_class.label))
                    hand_texts.append("- Score:%3.2f" % (hand_class.score * 100))
                # - テキスト表示に必要な座標など準備
                lm = hand_landmarks.landmark[0]
                lm_x = int(lm.x * img_w) - 50
                lm_y = int(lm.y * img_h) - 10
                lm_c = (64, 0, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # - テキスト出力
                for cnt, text in enumerate(hand_texts):
                    cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)

        # 画像の表示
        cv2.imshow("MediaPipe Hands", img)
        cv2.imwrite('hand.jpg', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            return NULL
    return res

def make_hand_landmarks_json(hand_landmarks):
    res = []
    index = 0
    for lm in (hand_landmarks.landmark):
        data_p = {}
        data = {}

        data_p['x'] = lm.x
        data_p['y'] = lm.y
        data_p['z'] = lm.z
        data['Index'] = index
        index += 1
        data['Point'] = data_p
        # break
        res.append(data)
    print(res)
    return res

def sendlog(num, string):
    now =  str(datetime.datetime.now(datetime.timezone.utc))
    logger.log(num, now + "> " + string)

def main():
    hands, cap = init_mp()
    try:
        time.sleep(0.5)
        data = GetHands(hands, cap)
        json_data = json.dumps(data)
        print(json_data)
        sendlog(30, json_data)
            
    except ConnectionAbortedError:
        print("Connection aborte")
    finally:
        cap.release()


if __name__ == "__main__":
    main()