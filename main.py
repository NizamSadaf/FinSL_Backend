# import os
# import tempfile
# import numpy as np
# import cv2
# import joblib
# import mediapipe as mp
# import tensorflow as tf
# from click import echo
#
# from fastapi import FastAPI, UploadFile, File, HTTPException
#
# # ------------------ CONFIG ------------------
#
# MODEL_PATH = "sign_model_tf213.h5"
# ENCODER_PATH = "label_encoder.pkl"
#
# CONF_THRESHOLD = 0.60
#
# app = FastAPI(title="FinSL Recognition API")
#
# model = None
# label_encoder = None
# MAX_LEN = None
# FEAT_DIM = 126
#
# # ------------------ CORS -----------------------
# from fastapi.middleware.cors import CORSMiddleware
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ------------------ MEDIAPIPE ------------------
#
# mp_hands = mp.solutions.hands
#
# def extract_hand_keypoints(video_path: str) -> np.ndarray:
#     cap = cv2.VideoCapture(video_path)
#     seq = []
#
#     with mp_hands.Hands(max_num_hands=2) as hands:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb)
#
#             left = np.zeros((21, 3), dtype=np.float32)
#             right = np.zeros((21, 3), dtype=np.float32)
#
#             if results.multi_hand_landmarks:
#                 for lm, hd in zip(results.multi_hand_landmarks,
#                                   results.multi_handedness):
#                     coords = np.array(
#                         [[p.x, p.y, p.z] for p in lm.landmark],
#                         dtype=np.float32
#                     )
#                     if hd.classification[0].label == "Left":
#                         left = coords
#                     else:
#                         right = coords
#
#             seq.append(np.concatenate([left.flatten(), right.flatten()]))
#
#     cap.release()
#     return np.array(seq, dtype=np.float32)
#
# # ------------------ STARTUP ------------------
#
# @app.on_event("startup")
# def load_artifacts():
#     global model, label_encoder, MAX_LEN
#
#     if not os.path.exists(MODEL_PATH):
#         raise RuntimeError("Model file not found")
#
#     if not os.path.exists(ENCODER_PATH):
#         raise RuntimeError("Label encoder not found")
#
#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         compile=False
#     )
#
#     label_encoder = joblib.load(ENCODER_PATH)
#
#     # infer MAX_LEN from model input
#     MAX_LEN = model.input_shape[1]
#
#     print("✅ Model loaded")
#     print("   Input shape:", model.input_shape)
#     print("   Classes:", label_encoder.classes_)
#
# # ------------------ ROUTES ------------------
#
#
# @app.get("/")
# def home():
#     return {"message": "server running"}
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model_loaded": model is not None,
#         "max_len": MAX_LEN,
#         "feat_dim": FEAT_DIM
#     }
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#
#     suffix = os.path.splitext(file.filename)[1]
#     if suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
#         raise HTTPException(status_code=400, detail="Unsupported video format")
#
#     tmp_path = os.path.join(
#         tempfile.gettempdir(),
#         f"upload_{os.getpid()}{suffix}"
#     )
#
#     with open(tmp_path, "wb") as f:
#         f.write(await file.read())
#
#     try:
#         seq = extract_hand_keypoints(tmp_path)
#
#         if len(seq) == 0:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No hand keypoints detected"
#             )
#
#         if len(seq) < MAX_LEN:
#             pad = np.zeros((MAX_LEN - len(seq), FEAT_DIM), dtype=np.float32)
#             seq = np.vstack([seq, pad])
#         else:
#             seq = seq[:MAX_LEN]
#
#         x = np.expand_dims(seq, axis=0)
#
#         probs = model.predict(x, verbose=0)[0]
#         idx = int(np.argmax(probs))
#         conf = float(probs[idx])
#
#         label = label_encoder.inverse_transform([idx])[0]
#
#         if conf < CONF_THRESHOLD:
#             label = "UNKNOWN"
#
#         return {
#             "predicted_label": label,
#             "confidence": conf,
#             "input_shape": list(x.shape)
#         }
#
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
# -------------------------2nd solution -----------------------
# import os
# import tempfile
# import numpy as np
# import cv2
# import joblib
# import mediapipe as mp
# import tensorflow as tf
#
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
#
# # ------------------ CONFIG ------------------
#
# MODEL_PATH = "sign_model_tf213_new.h5"
# ENCODER_PATH = "label_encoder_new.pkl"
#
# CONF_THRESHOLD = 0.5
#
# app = FastAPI(title="FinSL Recognition API")
#
# model = None
# label_encoder = None
# MAX_LEN = None
# FEAT_DIM = 126
#
# # ------------------ CORS -----------------------
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # ------------------ MEDIAPIPE ------------------
#
# mp_hands = mp.solutions.hands
#
# def extract_hand_keypoints(video_path: str) -> np.ndarray:
#     cap = cv2.VideoCapture(video_path)
#     seq = []
#
#     with mp_hands.Hands(max_num_hands=2) as hands:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb)
#
#             left = np.zeros((21, 3), dtype=np.float32)
#             right = np.zeros((21, 3), dtype=np.float32)
#
#             if results.multi_hand_landmarks:
#                 for lm, hd in zip(results.multi_hand_landmarks,
#                                   results.multi_handedness):
#                     coords = np.array(
#                         [[p.x, p.y, p.z] for p in lm.landmark],
#                         dtype=np.float32
#                     )
#                     if hd.classification[0].label == "Left":
#                         left = coords
#                     else:
#                         right = coords
#
#             seq.append(np.concatenate([left.flatten(), right.flatten()]))
#
#     cap.release()
#     return np.array(seq, dtype=np.float32)  # shape: (T, 126)
#
# def to_fixed_length(seq: np.ndarray, T: int, feat_dim: int = 126) -> np.ndarray:
#     """
#     Match training behavior:
#     - if t < T: zero-pad
#     - if t > T: uniform sampling
#     - if t == T: return as-is
#     """
#     t = seq.shape[0]
#
#     if t == 0:
#         return np.zeros((T, feat_dim), dtype=np.float32)
#
#     if t == T:
#         return seq.astype(np.float32)
#
#     if t > T:
#         idx = np.linspace(0, t - 1, T).astype(int)
#         return seq[idx].astype(np.float32)
#
#     # t < T
#     pad = np.zeros((T - t, feat_dim), dtype=np.float32)
#     return np.vstack([seq.astype(np.float32), pad])
#
# # ------------------ STARTUP ------------------
#
# @app.on_event("startup")
# def load_artifacts():
#     global model, label_encoder, MAX_LEN
#
#     if not os.path.exists(MODEL_PATH):
#         raise RuntimeError("Model file not found")
#
#     if not os.path.exists(ENCODER_PATH):
#         raise RuntimeError("Label encoder not found")
#
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     label_encoder = joblib.load(ENCODER_PATH)
#
#     # infer MAX_LEN from model input
#     MAX_LEN = model.input_shape[1]
#
#     print("✅ Model loaded")
#     print("   Input shape:", model.input_shape)
#     print("   Classes:", label_encoder.classes_)
#
# # ------------------ ROUTES ------------------
#
# @app.get("/")
# def home():
#     return {"message": "server running"}
#
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "model_loaded": model is not None,
#         "max_len": MAX_LEN,
#         "feat_dim": FEAT_DIM
#     }
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")
#
#     suffix = os.path.splitext(file.filename)[1]
#     if suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
#         raise HTTPException(status_code=400, detail="Unsupported video format")
#
#     tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{os.getpid()}{suffix}")
#
#     with open(tmp_path, "wb") as f:
#         f.write(await file.read())
#
#     try:
#         seq = extract_hand_keypoints(tmp_path)
#
#         if seq.shape[0] == 0:
#             raise HTTPException(status_code=400, detail="No hand keypoints detected")
#
#         # ✅ fixed-length standardization using uniform sampling (matches training)
#         seq = to_fixed_length(seq, MAX_LEN, FEAT_DIM)
#
#         x = np.expand_dims(seq, axis=0)  # (1, MAX_LEN, 126)
#
#         probs = model.predict(x, verbose=0)[0]
#         idx = int(np.argmax(probs))
#         conf = float(probs[idx])
#
#         label = label_encoder.inverse_transform([idx])[0]
#
#         if conf < CONF_THRESHOLD:
#             label = "UNKNOWN"
#
#         return {
#             "predicted_label": label,
#             "confidence": conf,
#             "input_shape": list(x.shape)
#         }
#
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

#  ---------------- 3rd Solution -------------

import os
import tempfile
import numpy as np
import cv2
import joblib
import mediapipe as mp
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ------------------ CONFIG ------------------

MODEL_PATH = "sign_model_tf213_new.h5"
ENCODER_PATH = "label_encoder_new.pkl"

CONF_THRESHOLD = 0.5          # top-1 confidence threshold
MARGIN_THRESHOLD = 0.15       # top1-top2 margin threshold (helps reduce ambiguous wrong outputs)
SLIDE_STEP = 10               # sliding window step for long videos (5 = more accurate, 10 = faster)

app = FastAPI(title="FinSL Recognition API")

model = None
label_encoder = None
MAX_LEN = None
FEAT_DIM = 126

# ------------------ CORS -----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://fin-sl-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ MEDIAPIPE ------------------

mp_hands = mp.solutions.hands

def extract_hand_keypoints(video_path: str) -> np.ndarray:
    """
    Returns: (T, 126) where 126 = left(21x3) + right(21x3).
    Missing hand in a frame -> zeros for that hand.
    """
    cap = cv2.VideoCapture(video_path)
    seq = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left = np.zeros((21, 3), dtype=np.float32)
            right = np.zeros((21, 3), dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    if hd.classification[0].label == "Left":
                        left = coords
                    else:
                        right = coords

            seq.append(np.concatenate([left.flatten(), right.flatten()]))

    cap.release()
    return np.array(seq, dtype=np.float32)  # (T, 126)


def normalize_seq(seq_126: np.ndarray) -> np.ndarray:
    """
    Match training preprocessing:
    - wrist-centered per hand (landmark 0)
    - scale-normalized by distance wrist -> middle_mcp (landmark 9)
    Input:  (T, 126)
    Output: (T, 126)
    """
    if seq_126.shape[0] == 0:
        return seq_126

    seq = seq_126.reshape((-1, 2, 21, 3)).astype(np.float32)  # (T, hand, lm, xyz)
    out = np.zeros_like(seq, dtype=np.float32)

    for h in range(2):
        hand = seq[:, h, :, :]          # (T, 21, 3)
        wrist = hand[:, 0:1, :]         # (T, 1, 3)
        hand0 = hand - wrist            # center at wrist

        scale = np.linalg.norm(hand0[:, 9, :], axis=1, keepdims=True)  # (T,1)
        scale = np.clip(scale, 1e-6, None)

        handn = hand0 / scale[:, None, :]  # (T,21,3)
        out[:, h, :, :] = handn

    return out.reshape((-1, 126)).astype(np.float32)


def to_fixed_length(seq: np.ndarray, T: int, feat_dim: int = 126) -> np.ndarray:
    """
    Standardize length:
    - if t < T: zero-pad
    - if t > T: uniform sampling across the full sequence
    - if t == T: return as-is
    """
    t = seq.shape[0]

    if t == 0:
        return np.zeros((T, feat_dim), dtype=np.float32)

    if t == T:
        return seq.astype(np.float32)

    if t > T:
        idx = np.linspace(0, t - 1, T).astype(int)
        return seq[idx].astype(np.float32)

    pad = np.zeros((T - t, feat_dim), dtype=np.float32)
    return np.vstack([seq.astype(np.float32), pad])


def predict_sliding_window(
    seq: np.ndarray,
    T: int,
    step: int = 10,
    threshold: float = 0.7,
    margin: float = 0.15
):
    """
    Sliding-window inference for long uploads.
    - If t <= T: predict once on standardized sequence.
    - If t > T: slide window (T) and choose best-confidence window.
    Unknown rule: top1<threshold OR (top1-top2)<margin
    Returns: (label, confidence)
    """
    t = seq.shape[0]
    if t == 0:
        return ("UNKNOWN", 0.0)

    # Short: single prediction
    if t <= T:
        win = to_fixed_length(seq, T, FEAT_DIM)
        x = np.expand_dims(win, axis=0)  # (1, T, 126)
        probs = model.predict(x, verbose=0)[0]
        top1 = int(np.argmax(probs))
        top1c = float(probs[top1])
        top2 = int(np.argsort(probs)[-2])
        top2c = float(probs[top2])

        label = label_encoder.inverse_transform([top1])[0]
        if (top1c < threshold) or ((top1c - top2c) < margin):
            return ("UNKNOWN", top1c)
        return (label, top1c)

    # Long: sliding window
    best_probs = None
    best_conf = -1.0

    for start in range(0, t - T + 1, step):
        win = seq[start:start + T]  # (T,126)
        x = np.expand_dims(win, axis=0)
        probs = model.predict(x, verbose=0)[0]
        conf = float(np.max(probs))
        if conf > best_conf:
            best_conf = conf
            best_probs = probs

    top1 = int(np.argmax(best_probs))
    top1c = float(best_probs[top1])
    top2 = int(np.argsort(best_probs)[-2])
    top2c = float(best_probs[top2])

    label = label_encoder.inverse_transform([top1])[0]
    if (top1c < threshold) or ((top1c - top2c) < margin):
        return ("UNKNOWN", top1c)
    return (label, top1c)


# ------------------ STARTUP ------------------

@app.on_event("startup")
def load_artifacts():
    global model, label_encoder, MAX_LEN

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Label encoder not found: {ENCODER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    label_encoder = joblib.load(ENCODER_PATH)

    MAX_LEN = model.input_shape[1]

    print("Model loaded")
    print("Input shape:", model.input_shape)
    print("MAX_LEN:", MAX_LEN)
    print("Classes:", getattr(label_encoder, "classes_", []))


# ------------------ ROUTES ------------------
@app.get("/")
def home():
    return {"message": "server running"}
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "max_len": MAX_LEN,
        "feat_dim": FEAT_DIM,
        "conf_threshold": CONF_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "slide_step": SLIDE_STEP
    }
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    suffix = os.path.splitext(file.filename)[1]
    if suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{os.getpid()}{suffix}")

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        # 1) Extract raw keypoints (T,126)
        seq = extract_hand_keypoints(tmp_path)
        if seq.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No hand keypoints detected")

        # 2) Normalize
        seq = normalize_seq(seq)
        # 3) Sliding window inference
        label, conf = predict_sliding_window(
            seq=seq,
            T=MAX_LEN,
            step=SLIDE_STEP,
            threshold=CONF_THRESHOLD,
            margin=MARGIN_THRESHOLD
        )
        return {
            "predicted_label": label,
            "confidence": conf,
            "max_len": MAX_LEN,
            "feat_dim": FEAT_DIM
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



