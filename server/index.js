
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const upload = multer({ dest: path.join(__dirname, 'uploads') });

const ROOT = path.join(__dirname, '..');
const MODELS_DIR = path.join(ROOT, 'models');
const OUTPUTS_DIR = path.join(ROOT, 'outputs');

app.use(cors());
app.use(express.json());

app.get('/api/status', (req, res) => {
  const modelPath = path.join(MODELS_DIR, 'nb_model.pkl');
  const exists = fs.existsSync(modelPath);
  res.json({
    model_path: modelPath,
    model_found: exists,
    tip: exists ? null : 'Train with: python model_utils.py (writes models/nb_model.pkl)'
  });
});

app.post('/api/analyze', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).send('No file uploaded');
  const ticket = uuidv4();
  const py = spawn('python', [
    path.join(__dirname, 'modelProcessor.py'),
    '--input', req.file.path,
    '--outputs', OUTPUTS_DIR,
    '--models', MODELS_DIR,
    '--ticket', ticket
  ], { cwd: ROOT });

  let out = '', err = '';
  py.stdout.on('data', d => out += d.toString());
  py.stderr.on('data', d => err += d.toString());
  py.on('close', (code) => {
    fs.unlink(req.file.path, () => {}); // cleanup
    if (code !== 0) {
      return res.status(500).send(err || 'Python scorer failed');
    }
    try {
      const payload = JSON.parse(out);
      res.json(payload);
    } catch (e) {
      // Attempt to recover if the scorer printed non-JSON noise before/after the JSON.
      // Try to extract the first JSON object from stdout.
      const first = out.indexOf('{');
      const last = out.lastIndexOf('}');
      if (first !== -1 && last !== -1 && last > first) {
        const sub = out.substring(first, last + 1);
        try {
          const payload = JSON.parse(sub);
          return res.json(payload);
        } catch (e2) {
          return res.status(500).send('Failed to parse scorer output (recovery failed): ' + e2.message + '\n' + out + '\nSTDERR:\n' + err);
        }
      }
      return res.status(500).send('Failed to parse scorer output: ' + e.message + '\n' + out + '\nSTDERR:\n' + err);
    }
  });
});

app.get('/api/download/:ticket', (req, res) => {
  const fp = path.join(OUTPUTS_DIR, `result_${req.params.ticket}.csv`);
  if (!fs.existsSync(fp)) return res.status(404).send('File not found');
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename=result_${req.params.ticket}.csv`);
  fs.createReadStream(fp).pipe(res);
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT}`);
});
