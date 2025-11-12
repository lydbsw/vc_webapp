import React from 'react';
import { Bar, Scatter, Bubble, Chart } from 'react-chartjs-2';
import ChartJS from '../chart/setup';
import './ChartsPanel.css';

function buildTopkData(res) {
  if (!res?.topk) return null;
  const rows = res.preview_rows || [];
  const fundingMap = Object.fromEntries(rows.map(r => [r._key_name, Number(r._last_round_amount_usd) || 0]));
  const points = res.topk.labels.map((name, i) => {
    const score = Number(res.topk.values[i]) || 0;
    const funding = fundingMap[name] || 0;
    const r = Math.max(4, Math.sqrt(funding || 0) / 1000);
    return { x: score, y: funding || 0, r };
  });
  return { datasets: [{ label: 'Top companies', data: points, backgroundColor: 'rgba(77,163,255,0.85)', borderColor: 'rgba(77,163,255,1)' }] };
}

function buildSectorTree(res) {
  if (!res?.sector) return null;
  const labels = res.sector.labels || [];
  const values = res.sector.values || [];
  const tree = labels.map((lab, i) => ({ v: values[i] || 0, name: lab }));
  return { datasets: [{ tree, key: 'v', groups: ['name'] }] };
}

export default function ChartsPanel({ res }) {
  const topkData = buildTopkData(res);
  const sectorTreeData = buildSectorTree(res);
  const histData = res?.hist ? {
    labels: res.hist.edges.slice(1).map((e,i)=>`${res.hist.edges[i].toFixed(2)}–${e.toFixed(2)}`),
    datasets: [{ label: 'Count', data: res.hist.counts, backgroundColor: 'rgba(0, 0, 0, 1)', borderColor: 'rgba(255, 255, 255, 1)', borderWidth: 1 }]
  } : null;
  const scatterData = res?.scatter ? { datasets: [{ label: 'Amount vs Score', data: res.scatter.x.map((x,i)=>({x, y: res.scatter.y[i]})), pointRadius: 2 }] } : null;

  return (
    <section className="charts-panel-root">
      <div className="chart-card">
        <h4>Score Distribution</h4>
        {histData ? <Bar data={histData} /> : <div>No data</div>}
      </div>
      <div className="chart-card">
        <h4>Top 20 (score vs funding)</h4>
        {topkData ? (
          <Bubble data={topkData} options={{ scales: { x: { title: { display: true, text: 'Score' }, min: 0, max: 1 }, y: { title: { display: true, text: 'Last round amount (USD)' }, type: 'logarithmic' } }, plugins: { tooltip: { callbacks: { label: (ctx) => { const d = ctx.raw || {}; return `${d.x?.toFixed(3) || ''} — $${(d.y || 0).toLocaleString()}`; } } }, legend: { display: false } } }} />
        ) : <div>No data</div>}
      </div>
      <div className="chart-card">
        <h4>Companies by sector</h4>
        {sectorTreeData ? (
          <Chart type="treemap" data={sectorTreeData} options={{ plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => { const v = ctx.dataset?.tree?.[ctx.dataIndex]?.v ?? ctx.raw?.v ?? ctx.parsed?.v; const n = ctx.dataset?.tree?.[ctx.dataIndex]?.name ?? ctx.label; return `${n}: ${v}`; } } } } }} />
        ) : <div>No data</div>}
      </div>
      <div className="chart-card">
        <h4>Score vs last round amount</h4>
        {scatterData ? <Scatter data={scatterData} options={{scales:{x:{type:'logarithmic'}}}} /> : <div>No data</div>}
      </div>
    </section>
  );
}
