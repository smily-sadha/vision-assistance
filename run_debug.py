import traceback
import numpy as np
from utils.face_utils import build_faiss_index

print('Loading DB')
try:
    db = np.load('embeddings/face_db_ssd.npz')
    print('db loaded, dim', db['embeddings'].shape)
except Exception as e:
    print('DB load error', e)

print('Building index')
try:
    index = build_faiss_index(db['embeddings'].astype('float32'))
    print('index ok')
except Exception as e:
    print('Index error', e)

print('Loading SSD detector')
import cv2
try:
    prototxt = 'model/deploy.prototxt'
    model = 'model/res10_300x300_ssd_iter_140000.caffemodel'
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
    print('detector ok')
except Exception as e:
    print('detector error', e)

print('Loading insightface')
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1)
    embedder = app.models['recognition']
    print('insightface loaded, models:', list(app.models.keys()))
except Exception as e:
    print('insightface error')
    traceback.print_exc()

print('Loading YOLO')
try:
    from ultralytics import YOLO
    yolo = YOLO('assets/models/yolo/yolov8n.pt')
    print('yolo loaded')
except Exception:
    print('yolo error')
    traceback.print_exc()

print('Initializing audio engines')
try:
    from vision.audio.tts.edge_tts_engine import TTSEngine
    from vision.audio.stt.stt_vad_engine import STTVADEngine

    tts = TTSEngine()
    stt = STTVADEngine(silence_timeout=1.0)
    print('audio engines initialized')
except Exception:
    print('audio init error')
    traceback.print_exc()
