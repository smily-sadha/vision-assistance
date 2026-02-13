import os
import urllib.request

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)

print("[INFO] Downloading face detection and embedding models...")

# 1. Download deploy.prototxt
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
prototxt_path = "model/deploy.prototxt"

if not os.path.exists(prototxt_path):
    print(f"Downloading {prototxt_path}...")
    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    print("✓ Downloaded deploy.prototxt")
else:
    print("✓ deploy.prototxt already exists")

# 2. Download caffemodel
model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(model_path):
    print(f"Downloading {model_path} (this may take a few minutes)...")
    urllib.request.urlretrieve(model_url, model_path)
    print("✓ Downloaded res10_300x300_ssd_iter_140000.caffemodel")
else:
    print("✓ res10_300x300_ssd_iter_140000.caffemodel already exists")

# 3. Download OpenFace embedding model
openface_url = "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7"
openface_path = "openface_nn4.small2.v1.t7"

if not os.path.exists(openface_path):
    print(f"Downloading {openface_path} (this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(openface_url, openface_path)
        print("✓ Downloaded openface_nn4.small2.v1.t7")
    except Exception as e:
        print(f"⚠ Failed to download from primary source: {e}")
        print("Trying alternative source...")
        # Alternative source
        alt_url = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"
        try:
            urllib.request.urlretrieve(alt_url, openface_path)
            print("✓ Downloaded openface_nn4.small2.v1.t7 from alternative source")
        except Exception as e2:
            print(f"✗ Failed to download OpenFace model: {e2}")
            print("\nPlease download manually from:")
            print("https://github.com/cmusatyalab/openface/blob/master/models/openface/nn4.small2.v1.t7")
else:
    print("✓ openface_nn4.small2.v1.t7 already exists")

print("\n[SUCCESS] Model download complete!")
print("\nModel files:")
print(f"  - {prototxt_path}")
print(f"  - {model_path}")
print(f"  - {openface_path}")