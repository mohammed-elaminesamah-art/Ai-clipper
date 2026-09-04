---
title: AI Viral Video Clipper
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# AI Viral Video Clipper

Turn long videos into short-form clips optimized for retention and viral potential.

## Deployment on Streamlit Cloud (FREE)

Since Hugging Face Spaces now require a PRO subscription for Gradio apps, we use **Streamlit Community Cloud** – completely free and supports Python + FFmpeg.

### Steps to Deploy (from your Android phone)

1. **Create a GitHub account** (if you don't have one) – [github.com/join](https://github.com/join)

2. **Create a new repository** named `ai-viral-clipper` and upload all files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
   - `clipper/` folder with all `.py` files

3. **Sign up for Streamlit Cloud** at [share.streamlit.io](https://share.streamlit.io) using your GitHub account.

4. **Deploy the app:**
   - Click **"New app"**
   - Select your repository and branch
   - Main file: `app.py`
   - Click **"Deploy"**

5. **Add Secrets (optional):**
   - Go to your app's settings on Streamlit Cloud
   - Add `HF_TOKEN` as a secret
   - You can get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

6. **Wait for the build** – Streamlit will install dependencies and FFmpeg (via `packages.txt`).

7. **Open your app** – share the link and use it from your phone.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | No | Hugging Face token for LLM ranking (fallback to heuristics if not set) |

### Limitations

- **CPU‑only** – Whisper large‑v3‑turbo runs on CPU, so a 1‑hour video may take 15–30 minutes.
- **Memory** – Streamlit Cloud offers ~16GB RAM (enough for most videos).
- **Timeout** – Streamlit apps have no hard timeout (unlike Hugging Face Spaces), so longer videos are possible.
- **Free tier** – Public apps only; private apps require a paid plan.

### Troubleshooting

- **Build fails** – ensure `packages.txt` contains `ffmpeg`.
- **Transcription slow** – try shorter videos first.
- **No clips generated** – video may have little speech or poor audio quality.
- **FFmpeg errors** – check if your video is corrupted or unsupported.

### Credits

Based on [clipper](https://github.com/Minecraftpro546/clipper) by Minecraftpro546. Adapted for Streamlit and server‑side LLM inference.

---

**This app is completely free to use and runs on Streamlit Cloud.**
