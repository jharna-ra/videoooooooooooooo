"""HTTP API around pipeline.py. Run on a GPU machine:  uvicorn server:app --host 0.0.0.0 --port 8000"""
import os, re, glob, uuid, base64, threading, cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pipeline as pl

DATA = os.environ.get('DATA_DIR', '/tmp/remover')
KEY = os.environ.get('API_KEY', '')          # optional shared secret (set it on a public server!)
os.makedirs(DATA, exist_ok=True)


def auth(x_api_key: str = Header(default=''), key: str = Query(default='')):
    if KEY and KEY not in (x_api_key, key):
        raise HTTPException(401, 'Bad API key')


app = FastAPI(dependencies=[Depends(auth)])
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
JOBS, LOCK = {}, threading.Lock()            # one GPU job at a time


class P(BaseModel):
    id: str
    x: int = 900
    y: int = 700
    w: int = 380
    h: int = 300
    thr: int = 200
    thick: int = 9
    grow: int = 2
    at: float = 5
    start: float = 0
    end: float = 0
    mode: str = 'test'


def vpath(vid):
    f = glob.glob(f"{DATA}/{re.sub('[^a-f0-9]', '', vid)}/input.*")
    if not f:
        raise HTTPException(404, 'Video not found, upload again')
    return f[0]


def b64(img, ext):
    return f"data:image/{ext};base64," + base64.b64encode(cv2.imencode('.' + ext, img)[1]).decode()


@app.get('/health')
def health():
    return {'ok': True}


@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:12]
    os.makedirs(f'{DATA}/{vid}')
    ext = re.sub('[^.a-z0-9]', '', os.path.splitext(file.filename or '')[1].lower())[:6] or '.mp4'
    path = f'{DATA}/{vid}/input{ext}'
    with open(path, 'wb') as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    try:
        fps, _, dur = pl.probe(path)
    except Exception:
        raise HTTPException(400, 'Not a readable video')
    return {'id': vid, 'fps': fps, 'duration': dur}


@app.post('/preview')
def preview(p: P):
    fr = pl.grab(vpath(p.id), p.at)
    if fr is None:
        raise HTTPException(400, 'Could not read a frame at that time')
    vis, crop, pct = pl.draw_preview(fr, (p.x, p.y, p.w, p.h), p.thr, p.thick, p.grow)
    return {'frame': b64(vis, 'jpg'), 'crop': b64(crop, 'png'), 'pct': pct, 'size': [fr.shape[1], fr.shape[0]]}


def worker(job, p, video):
    with LOCK:
        job['status'] = 'running'
        try:
            def cb(d, t, e):
                job.update(done=d, total=t, eta=e)
            work = job['work']
            res = pl.run(video, work + '/out.mp4', work, (p.x, p.y, p.w, p.h), p.thr, p.thick, p.grow,
                         p.start, p.end or None, p.mode == 'test', cb=cb)
            if p.mode == 'test':
                cv2.imwrite(work + '/sample.png', res)
            job['status'] = 'done'
        except Exception as e:
            job.update(status='error', error=str(e))


@app.post('/jobs')
def start(p: P):
    video = vpath(p.id)
    jid = uuid.uuid4().hex[:12]
    work = f'{DATA}/{p.id}/{jid}'
    os.makedirs(work)
    JOBS[jid] = {'status': 'queued', 'done': 0, 'total': 0, 'eta': None, 'error': None, 'mode': p.mode, 'work': work}
    threading.Thread(target=worker, args=(JOBS[jid], p, video), daemon=True).start()
    return {'job': jid}


def job_or_404(jid):
    if jid not in JOBS:
        raise HTTPException(404, 'Unknown job')
    return JOBS[jid]


@app.get('/jobs/{jid}')
def status(jid: str):
    return {k: v for k, v in job_or_404(jid).items() if k != 'work'}


@app.get('/jobs/{jid}/sample')
def sample(jid: str):
    return FileResponse(job_or_404(jid)['work'] + '/sample.png', media_type='image/png')


@app.get('/jobs/{jid}/result')
def result(jid: str):
    return FileResponse(job_or_404(jid)['work'] + '/out.mp4', media_type='video/mp4', filename='cleaned.mp4')
