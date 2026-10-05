"""Run the backend inside Colab (T4 GPU) and print a public https URL. Usage: !python colab_launch.py"""
import os, sys, re, time, threading, subprocess
os.environ.setdefault('PROPAINTER_DIR', '/content/ProPainter')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvicorn
from server import app

threading.Thread(target=lambda: uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning'), daemon=True).start()
cf = '/content/cloudflared'
if not os.path.exists(cf):
    subprocess.run(f'wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O {cf} && chmod +x {cf}', shell=True, check=True)
p = subprocess.Popen([cf, 'tunnel', '--url', 'http://localhost:8000'], stderr=subprocess.PIPE, text=True)
for line in p.stderr:
    m = re.search(r'https://[-a-z0-9]+\.trycloudflare\.com', line)
    if m:
        print('\nBACKEND URL (paste into the website):', m.group(0), '\n', flush=True)
        break
threading.Thread(target=lambda: [None for _ in p.stderr], daemon=True).start()
while True:
    time.sleep(3600)
