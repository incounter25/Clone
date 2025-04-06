import streamlit as st
import cv2
import numpy as np
from PIL import Image
import insightface
import joblib
import time

st.title("📸 실시간 얼굴 판별기")
st.write("웹캠을 통해 얼굴을 실시간으로 탐지하고 사람이 맞는 경우 표시합니다.")

# 모델 불러오기
face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)
clf = joblib.load("is_human_classifier.pkl")

# 웹캠 시작
run = st.checkbox('▶️ 웹캠 시작')

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)
    st.write("⏳ 웹캠에서 프레임을 수신 중...")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("웹캠에서 프레임을 가져오지 못했습니다.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_model.get(frame)

        human_count = 1
        for i, face in enumerate(faces):
            emb = face.embedding.reshape(1, -1)
            pred = clf.predict(emb)[0]
            proba = clf.predict_proba(emb)[0][pred]
            is_human = (pred == 1)
            label = f"{human_count}. Human" if is_human else "UnHuman"

            x1, y1, x2, y2 = map(int, face.bbox)
            color = (255, 0, 0) if is_human else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if is_human:
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                human_count += 1

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.01)  # 프레임 속도 조절

else:
    if cap:
        cap.release()
    FRAME_WINDOW.image(np.zeros((480, 640, 3), dtype=np.uint8))
    st.write("⏹️ 웹캠이 정지되었습니다.")
