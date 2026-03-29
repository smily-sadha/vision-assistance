from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["frame"]
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # show Raspberry Pi camera on laptop
    cv2.imshow("Raspberry Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC pressed")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = int(gray.mean())

    return jsonify({"message": f"Frame processed. brightness={brightness}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)