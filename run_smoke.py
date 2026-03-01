import subprocess, sys
result = subprocess.run(
    [sys.executable, 'test_voice_smoke.py'],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr[-800:] if len(result.stderr) > 800 else result.stderr)
print("Return code:", result.returncode)
