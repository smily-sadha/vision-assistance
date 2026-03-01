import pyaudio
import numpy as np

CHUNK = 4096
RATE = 16000

pa = pyaudio.PyAudio()

print("Available audio input devices:")
for i in range(pa.get_device_count()):
    devinfo = pa.get_device_info_by_index(i)
    if devinfo['maxInputChannels'] > 0:
        print(f"  [{i}] {devinfo['name']}")

default_in = pa.get_default_input_device_info()
print(f"\nDefault input device: [{default_in['index']}] {default_in['name']}")

try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print("\nListening for 5 seconds. PLEASE SPEAK loudly into the microphone...")
    max_rms = 0
    for _ in range(int(RATE / CHUNK * 5)):
        raw = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(raw, dtype=np.int16)
        rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        print(f"Current RMS: {rms}")
        if rms > max_rms:
            max_rms = rms
    print(f"\nMax RMS detected: {max_rms}")
    stream.stop_stream()
    stream.close()
except Exception as e:
    print(f"Microphone error: {e}")

pa.terminate()
