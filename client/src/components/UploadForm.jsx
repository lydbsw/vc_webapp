import React from 'react';
import { downloadUrl } from '../services/api';

export default function UploadForm({ file, onFileChange, onAnalyze, result }) {
  return (
    <section style={{padding:'16px 20px', display:'flex', gap:12, alignItems:'center', borderBottom:'1px solid #222'}}>
      <input type="file" accept=".csv" onChange={e=>onFileChange(e.target.files?.[0] || null)} />
      <button disabled={!file} onClick={onAnalyze} style={{background:'#4da3ff', color:'#000', border:'none', padding:'8px 14px', borderRadius:8, cursor:'pointer'}}>
        Analyze
      </button>
      {result && <a href={downloadUrl(result.ticket)} style={{marginLeft:12, color:'#4da3ff'}}>Download processed CSV</a>}
    </section>
  );
}
