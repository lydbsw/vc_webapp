
import React, { useState, useEffect } from 'react';
import './chart/setup';
import UploadForm from './components/UploadForm';
import PreviewTable from './components/PreviewTable';
import CompanyDetail from './components/CompanyDetail';
import ChartsPanel from './components/ChartsPanel';
import api from './services/api';

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [res, setRes] = useState(null);
  const [selectedCompany, setSelectedCompany] = useState(null);

  useEffect(() => {
    api.fetchStatus().then(j => {
      setStatus(j.model_found ? `Model loaded: ${j.model_path}` : `No model found at ${j.model_path}. Train via: python model_utils.py`);
    }).catch(()=> setStatus('Cannot reach server'));
  }, []);

  const onAnalyze = async () => {
    if (!file) return;
    try {
      const data = await api.analyzeFile(file);
      setRes(data);
    } catch (err) {
      alert(err.message || 'Analyze failed');
    }
  };

  const kpi = (label, value) => (
    <div style={{background:'#15171c',padding:12,borderRadius:12,minWidth:140}}>
      <div style={{color:'#ffffffff',fontSize:12}}>{label}</div>
      <div style={{fontSize:20,fontWeight:600}}>{value}</div>
    </div>
  );

  return (
    <div style={{color:'#969696ff', background:'#0b0c10', minHeight:'100vh', fontFamily:'ui-sans-serif, system-ui'}}>
      <header style={{padding:'24px 20px', borderBottom:'1px solid #222'}}>
        <h2 style={{margin:0}}>VC Analyzer — Upload & Score</h2>
        <p style={{margin:0, color:'#ffffffff'}}>{status}</p>
      </header>

      <UploadForm file={file} onFileChange={setFile} onAnalyze={onAnalyze} result={res} />

      {res && (
        <>
          <section style={{padding:'16px 20px', display:'flex', gap:18, flexWrap:'wrap'}}>
            {kpi('Rows', res.kpis?.rows ?? '—')}
            {kpi('Columns', res.kpis?.cols ?? '—')}
            {kpi('Model', res.kpis?.model ?? '—')}
          </section>

          <ChartsPanel res={res} />

          <section style={{padding:'16px 20px'}}>
            <h3>Preview</h3>
            <div style={{display:'flex', gap:12, alignItems:'flex-start'}}>
              <PreviewTable columns={res.preview_columns||[]} rows={res.preview_rows||[]} selectedCompany={selectedCompany} onSelect={setSelectedCompany} />
              <CompanyDetail company={selectedCompany} onClose={() => setSelectedCompany(null)} />
            </div>
          </section>
        </>
      )}
    </div>
  );
}
