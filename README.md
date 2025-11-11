
# VC Analyzer — React (client) + Express (server)
Structure mirrors `client/` and `server/` like the Northeastern-Superstore repo.

## Quick start
1) Train the model via `model_utils.py` (writes `models/nb_model.pkl`)
```bash
export VC_DATA_DIR=/absolute/path/to/dataset
python model_utils.py
```
2) Start server
```bash
cd server && npm install
node index.js   # http://localhost:5001
```
3) Start client
```bash
cd client && npm install
npm start       # http://localhost:3000
```
If running separately, set `client/.env` with `REACT_APP_API_BASE=http://localhost:5001`.
