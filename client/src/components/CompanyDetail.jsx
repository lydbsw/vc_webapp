import React from 'react';
import { Bar } from 'react-chartjs-2';
import './CompanyDetail.css';

export default function CompanyDetail({ company, onClose }) {
  if (!company) return <div className="company-empty">Click a row to view details</div>;

  const labels = ['Last round ($)','Investors','Articles','Days since'];
  const values = [
    Number(company._last_round_amount_usd) || 0,
    Number(company._num_investors) || 0,
    Number(company._num_articles) || 0,
    Number(company._days_since_last_round) || 0,
  ];
  const data = { labels, datasets:[{ label: company._key_name, data: values, backgroundColor: 'rgba(77,163,255,0.85)', borderColor: 'rgba(77,163,255,1)', borderWidth: 1 }] };

  return (
    <div className="company-detail-root">
      <div className="company-sticky">
        <div className="company-card">
          <div className="company-header">
            <h4 className="company-title">{company._key_name}</h4>
            <button onClick={onClose} className="company-close-btn">✕</button>
          </div>
          <div className="company-info">
            <div><strong>Sector:</strong> {company._sector || '—'}</div>
            <div><strong>Location:</strong> {company._location || '—'}</div>
            <div><strong>Last round:</strong> {company._last_round_amount_usd || '—'}</div>
            <div><strong>Investors:</strong> {company._num_investors || '—'}</div>
            <div><strong>Articles:</strong> {company._num_articles || '—'}</div>
            <div><strong>Days since last round:</strong> {company._days_since_last_round || '—'}</div>
          </div>
        </div>

        <div className="company-card company-metrics">
          <h4>Company metrics</h4>
          <Bar data={data} options={{plugins:{legend:{display:false}, tooltip:{enabled:true}}, scales:{y:{beginAtZero:true, ticks:{color:'#ddd'}}, x:{ticks:{color:'#ddd'}}}}} />
        </div>
      </div>
    </div>
  );
}
