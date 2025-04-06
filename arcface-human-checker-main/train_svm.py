import os
import cv2
import numpy as np
import insightface
from sklearn.svm import SVC
import joblib
from tqdm import tqdm


model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

X = []
y = []
def extract_embeddings(folder_path, label):
    for fname in tqdm(os.listdir(folder_path), desc=f"{label} - {os.path.basename(folder_path)}"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = model.get(img)
        if faces:
            emb = faces[0].embedding
            X.append(emb)
            y.append(label)

extract_embeddings("dataset/human", 1)
extract_embeddings("dataset/nonhuman", 0)

print(f"\n✅ 총 샘플 수: {len(X)}")
print(f"   - 사람 얼굴: {y.count(1)}")
print(f"   - 비사람: {y.count(0)}")


clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)


joblib.dump(clf, "is_human_classifier.pkl")
print("\n✅ 분류기 저장 완료: is_human_classifier.pkl")
