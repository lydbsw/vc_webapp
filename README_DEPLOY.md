# Deploying vc_webapp (Vercel client + Render server)

This document describes the recommended deployment: static React client on Vercel, server (Node + Python scorer) on Render using the provided `server/Dockerfile`.

## Prerequisites
- Project pushed to GitHub.
- Accounts on Vercel and Render connected to your GitHub.
- (Optional) Render paid plan if you need persistent disk for `outputs/` or large models.

---

## Server (Render) — Docker

1. Go to https://render.com and create a new **Web Service**.
2. Connect your GitHub repository and select the `vc_webapp` repository and `main` branch.
3. Choose **Docker** as the environment. Set the **Dockerfile Path** to `server/Dockerfile` and leave the build context at the repository root.
4. Start Command: `node index.js` (the Dockerfile builds Node + Python and installs requirements).
5. Environment variables (set in Render's Service > Environment):
   - `PORT` — leave default (Render sets it automatically), or set to `5001`.
   - `ALLOWED_ORIGIN` — set to your Vercel client URL (e.g. `https://your-client.vercel.app`) to restrict CORS.
   - Any other secrets (S3 credentials, model config) as needed.
6. (Optional) Attach persistent disk if you want `models/` and `outputs/` to persist across deploys. Otherwise include model files in the repo or use S3.
7. Create the service and wait for the build to complete. Note the public URL (e.g., `https://vc-server.onrender.com`).

Verify server after deploy:
```bash
curl https://<your-render-url>/api/status
```

---

## Client (Vercel)

1. Go to https://vercel.com and import the `vc_webapp` repository.
2. During import, set the **Root Directory** to `client` so Vercel builds the React app.
3. Build Command: `npm run build` (CRA default). Output Directory: `build`.
4. Environment Variables (Vercel Project Settings > Environment Variables):
   - `REACT_APP_API_BASE` = `https://<your-render-url>` (the server URL from Render).
5. Deploy. Vercel will give you a client URL (e.g., `https://vc-client.vercel.app`).

Test the deployed app by visiting the Vercel URL and performing a small analyze/upload or checking UI features.

---

## Notes and Production Tips
- CORS: server now reads `ALLOWED_ORIGIN` and uses it for CORS.
- Models: include models in `models/` if small, otherwise use S3 and set `VC_MODEL_DIR` or modify the server to download models at startup.
- Outputs: use persistent disk or upload outputs to object storage if you need longer-term storage.
- Performance: current server spawns Python for each analyze request; for higher traffic consider converting the Python scorer to a persistent FastAPI service.

## Local testing with Docker Compose
From the repo root:
```bash
docker-compose up --build
# client: http://localhost:3000
# server: http://localhost:5001/api/status
```

---

If you want, I can: (A) walk you through the Render & Vercel UI steps live, or (B) add a GitHub Actions workflow to automate builds/deploys. Which do you prefer?
