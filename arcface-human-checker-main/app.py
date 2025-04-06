import streamlit as st
import cv2
import numpy as np
from PIL import Image
import insightface
import joblib


face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)
clf = joblib.load("is_human_classifier.pkl")


def predict_faces_and_draw(image):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = face_model.get(img_bgr)

    if not faces:
        return image, ["❌ 얼굴이 감지되지 않았습니다."]

    results = []
    human_count = 1

    for i, face in enumerate(faces):
        emb = face.embedding.reshape(1, -1)
        pred = clf.predict(emb)[0]
        proba = clf.predict_proba(emb)[0][pred]
        is_human = (pred == 1)
        label = f"{human_count}. 사람" if is_human else "비사람"
        result_text = f"[{i+1}] {'✅ 사람' if is_human else '🧸 비사람'} (정확도: {proba:.2f})"
        results.append(result_text)


        x1, y1, x2, y2 = map(int, face.bbox)
        color = (0, 255, 0) if is_human else (0, 0, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        if is_human:
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            human_count += 1


    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(img_rgb)
    return annotated_image, results


st.title("👤 얼굴 판별기 (멀티 얼굴 + 번호 부여)")
st.write("이미지 속 모든 얼굴을 탐지하고 사람이 맞는 경우 번호를 부여합니다.")

uploaded_image = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("분석 중..."):
        annotated_img, predictions = predict_faces_and_draw(image)

    st.image(annotated_img, caption="예측 결과", use_column_width=True)
    st.markdown("### 📋 판별 결과")
    for res in predictions:
        st.write(res)
