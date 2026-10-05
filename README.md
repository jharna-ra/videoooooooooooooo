# White text remover (ProPainter)

Removes only the white text pixels from a video and rebuilds them from neighbouring frames.

```
Browser ──> frontend/  (static page, hosted on Vercel)
                │  HTTPS
                ▼
          backend/  (FastAPI + ProPainter, needs an NVIDIA GPU)
```

**Why two parts?** Vercel has no GPU, a ~250 MB size limit and short function timeouts, so ProPainter
(PyTorch + ffmpeg, minutes to hours per video) cannot run there. Vercel hosts the web page; the GPU work
runs on a backend you start in Colab or deploy with Docker.

## 1. Start the backend

### Option A: Google Colab (free T4, same as your notebook)
Runtime > Change runtime type > **T4 GPU**, then run in a cell:
```
!git clone -q https://github.com/YOUR_USER/YOUR_REPO /content/app
!git clone -q https://github.com/sczhou/ProPainter /content/ProPainter
!pip install -q -r /content/ProPainter/requirements.txt -r /content/app/backend/requirements.txt
%env API_KEY=choose-a-secret
!python /content/app/backend/colab_launch.py
```
It prints `BACKEND URL: https://....trycloudflare.com`. Keep the cell running. The URL changes each session.

### Option B: Docker on a GPU host (Hugging Face Spaces, RunPod, ...)
```
cd backend
docker build -t text-remover .
docker run --gpus all -p 7860:7860 -e API_KEY=choose-a-secret text-remover
```
On Hugging Face: create a Docker Space with a GPU, upload the `backend/` files, add `API_KEY` as a secret.
The first run downloads the ProPainter weights automatically.

## 2. Put it on GitHub
```
git init && git add . && git commit -m "text remover"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## 3. Deploy the page on Vercel
1. vercel.com > **Add New > Project** > import the repo.
2. Framework preset **Other**, leave Build Command empty. (`vercel.json` already sets output to `frontend/`.
   Alternatively set **Root Directory** to `frontend`.)
3. Deploy, open the URL, paste the **Backend URL** and your **API key**, click *Save & test connection*.

## 4. Use it
Upload video > set the box (x, y, width, height from the grid in the preview) > **Preview mask**
(red must be only the letters; tune brightness / thickness / expand) > **Test 40 frames** > **Render full video**.

Notes
- The backend URL must be `https` (Vercel pages are https; browsers block mixed content).
- Set `API_KEY` on any public backend, otherwise anyone with the URL can use your GPU.
- Processes one job at a time. Uploaded files and results live in `DATA_DIR` (default `/tmp/remover`) and are lost on restart.
- For very long videos use Start/End to render in parts (e.g. 10 min each).
