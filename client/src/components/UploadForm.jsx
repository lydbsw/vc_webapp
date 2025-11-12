import React from 'react';
import { downloadUrl } from '../services/api';
import './UploadForm.css';

export default function UploadForm({ file, onFileChange, onAnalyze, result }) {
  return (
    <section className="upload-section">
      <input type="file" accept=".csv" onChange={e=>onFileChange(e.target.files?.[0] || null)} />

      <div className="analyze-btn-wrapper">
        <button
          disabled={!file}
          onClick={onAnalyze}
          className="analyze-btn"
          aria-label="Analyze CSV"
        >
          Analyze
        </button>
        <span className="analyze-tooltip">Click to analyze the uploaded CSV</span>
      </div>

      {result && <a href={downloadUrl(result.ticket)} className="download-link">Download processed CSV</a>}
    </section>
  );
}
