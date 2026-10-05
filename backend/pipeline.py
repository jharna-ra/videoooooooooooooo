"""White-text remover core (ProPainter). Extracted from the original notebook, no globals."""
import subprocess, json, os, glob, shutil, time, numpy as np, cv2

PP = os.environ.get('PROPAINTER_DIR', '/app/ProPainter')


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def probe(path):
    o = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'stream=r_frame_rate:format=duration', '-of', 'json', path])
    j = json.loads(o)
    n, d = j['streams'][0]['r_frame_rate'].split('/')
    return float(n) / float(d), j['streams'][0]['r_frame_rate'], float(j['format']['duration'])


def grab(path, t):
    o = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', path, '-frames:v', '1',
                                 '-f', 'image2pipe', '-vcodec', 'png', '-'])
    return cv2.imdecode(np.frombuffer(o, np.uint8), cv2.IMREAD_COLOR)


def text_mask(img, box, thr, thick, grow):
    # only thin, bright, colourless strokes = text pixels. Big bright areas and lines are NOT touched.
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spread = img.max(2).astype(np.int16) - img.min(2).astype(np.int16)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thick + 1, 2 * thick + 1))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    strong = (g >= thr) & (spread <= 60) & (th >= 25)
    weak = (g >= thr - 45) & (spread <= 80) & (th >= 10)
    near = cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = (strong | (weak & near)).astype(np.uint8)
    if grow > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    x, y, w, h = box
    inside = np.zeros_like(m)
    inside[max(0, y):y + h, max(0, x):x + w] = 1
    return m & inside


def roi_geometry(W, H, box, pad=24, minsz=256):
    x, y, w, h = box

    def fit(a0, a1, L):
        n = max(minsz, a1 - a0)
        n = min(L // 8 * 8, (n + 7) // 8 * 8)
        c = (a0 + a1) // 2
        s = max(0, min(c - n // 2, L - n))
        return s, n
    rx, rw = fit(max(0, x - pad), min(W, x + w + pad), W)
    ry, rh = fit(max(0, y - pad), min(H, y + h + pad), H)
    return rx, ry, rw, rh


def draw_preview(fr, box, thr, thick, grow):
    """Returns (frame with grid+box, crop original|red mask, % of box covered)."""
    H, W = fr.shape[:2]
    vis = fr.copy()
    for gx in range(0, W, 100):
        cv2.line(vis, (gx, 0), (gx, H), (255, 255, 0), 1)
        cv2.putText(vis, str(gx), (gx + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    for gy in range(0, H, 100):
        cv2.line(vis, (0, gy), (W, gy), (255, 255, 0), 1)
        cv2.putText(vis, str(gy), (2, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    x, y, w, h = box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 3)
    m = text_mask(fr, box, thr, thick, grow)
    red = fr.copy()
    red[m > 0] = (0, 0, 255)
    a, b = max(0, y - 20), max(0, x - 20)
    crop = np.hstack([fr[a:y + h + 20, b:x + w + 20], red[a:y + h + 20, b:x + w + 20]])
    return vis, crop, round(100 * float(m.sum()) / max(1, w * h), 1)


def propainter(frames, masks, work):
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work + '/f')
    os.makedirs(work + '/m')
    for i, (f, m) in enumerate(zip(frames, masks)):
        cv2.imwrite(f'{work}/f/{i:05d}.png', f)
        cv2.imwrite(f'{work}/m/{i:05d}.png', m * 255)
    r = subprocess.run(f'cd {PP} && python inference_propainter.py --video {work}/f --mask {work}/m --output {work}/out '
                       f'--fp16 --save_frames --mask_dilation 2 --subvideo_length 80 --neighbor_length 10 --ref_stride 10',
                       shell=True, capture_output=True, text=True)
    outs = sorted(glob.glob(f'{work}/out/**/*.png', recursive=True))
    if r.returncode or len(outs) != len(frames):
        raise RuntimeError('ProPainter failed:\n' + (r.stderr or r.stdout)[-1500:])
    return [cv2.imread(p) for p in outs]


def roi_stream(path, start, n, rate, roi):
    rx, ry, rw, rh = roi
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(start), '-i', path, '-vf',
                          f'fps={rate},crop={rw}:{rh}:{rx}:{ry}', '-frames:v', str(n), '-f', 'rawvideo',
                          '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE, bufsize=rw * rh * 3 * 8)
    size = rw * rh * 3
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        yield np.frombuffer(b, np.uint8).reshape(rh, rw, 3).copy()
    p.stdout.close()
    p.wait()


def run(video, out, work, box, thr, thick, grow, start=0, end=None, test=False, chunk=300, overlap=8, cb=None):
    """test=True -> returns a comparison image (original | mask | result). Otherwise writes `out` and returns it."""
    os.makedirs(work, exist_ok=True)
    fps, rate, dur = probe(video)
    end = min(end or dur, dur)
    length = end - start
    total = max(1, round(length * fps))
    if test:
        total = min(total, 40)
    first = grab(video, start)
    H, W = first.shape[:2]
    rx, ry, rw, rh = roi = roi_geometry(W, H, box)
    bx = max(0, box[0] - rx)
    by = max(0, box[1] - ry)
    rbox = (bx, by, min(box[2], rw - bx), min(box[3], rh - by))
    clean, mask = work + '/roi_clean.mkv', work + '/roi_mask.mkv'
    wc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{rw}x{rh}',
                           '-r', rate, '-i', '-', '-c:v', 'ffv1', clean], stdin=subprocess.PIPE)
    wm = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{rw}x{rh}',
                           '-r', rate, '-i', '-', '-c:v', 'ffv1', '-pix_fmt', 'gray', mask], stdin=subprocess.PIPE)
    t0 = time.time()
    st = {'done': 0, 'tail': [], 'new': [], 'sample': None}

    def flush():
        items = st['tail'] + st['new']
        frs = [a for a, b in items]
        mks = [b for a, b in items]
        nt = len(st['tail'])
        out_f = propainter(frs, mks, work + '/pp') if any(mk.any() for mk in mks[nt:]) else frs
        for o, mk in list(zip(out_f, mks))[nt:]:
            wc.stdin.write(np.ascontiguousarray(o).tobytes())
            wm.stdin.write((mk * 255).astype(np.uint8).tobytes())
        if test:
            st['sample'] = (frs, mks, out_f)
        st['done'] += len(st['new'])
        st['tail'] = items[-overlap:]
        st['new'] = []
        f = st['done'] / max(time.time() - t0, 1e-6)
        if cb:
            cb(st['done'], total, (total - st['done']) / max(f, 1e-6) / 60)

    for fr in roi_stream(video, start, total, rate, roi):
        st['new'].append((fr, text_mask(fr, rbox, thr, thick, grow)))
        if len(st['new']) >= chunk:
            flush()
    if st['new']:
        flush()
    wc.stdin.close(); wm.stdin.close(); wc.wait(); wm.wait()
    if test:
        frs, mks, out_f = st['sample']
        i = min(len(frs) - 1, 20)
        red = frs[i].copy()
        red[mks[i] > 0] = (0, 0, 255)
        return np.hstack([frs[i], red, out_f[i]])
    fc = f'[0:v]fps={rate}[b];[2:v]format=gray[m];[1:v][m]alphamerge[c];[b][c]overlay={rx}:{ry}:format=auto:shortest=1[v]'
    sh('ffmpeg -v error -y -ss %s -t %s -i "%s" -i "%s" -i "%s" -filter_complex "%s" '
       '-map "[v]" -map 0:a? -c:v libx264 -preset veryfast -crf 17 -pix_fmt yuv420p -c:a aac -b:a 160k '
       '-movflags +faststart "%s"' % (start, length, video, clean, mask, fc, out))
    return out
