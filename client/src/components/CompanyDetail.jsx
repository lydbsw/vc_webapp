import React from 'react';
import { Bar } from 'react-chartjs-2';

export default function CompanyDetail({ company, onClose }) {
  if (!company) return <div style={{background:'#0f1720', padding:12, borderRadius:12, color:'#a8a8ad'}}>Click a row to view details</div>;

  const labels = ['Last round ($)','Investors','Articles','Days since'];
  const values = [
    Number(company._last_round_amount_usd) || 0,
    Number(company._num_investors) || 0,
    Number(company._num_articles) || 0,
    Number(company._days_since_last_round) || 0,
  ];
  const data = { labels, datasets:[{ label: company._key_name, data: values, backgroundColor: 'rgba(77,163,255,0.85)', borderColor: 'rgba(77,163,255,1)', borderWidth: 1 }] };

  return (
    <div style={{width:380, flex:'0 0 380px'}}>
      <div style={{position:'sticky', top:24}}>
        <div style={{background:'#15171c', padding:14, borderRadius:12}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
            <h4 style={{margin:0}}>{company._key_name}</h4>
            <button onClick={onClose} style={{background:'transparent', border:'none', color:'#a8a8ad', cursor:'pointer'}}>✕</button>
          </div>
          <div style={{color:'#a8a8ad', marginTop:8}}>
            <div><strong>Sector:</strong> {company._sector || '—'}</div>
            <div><strong>Location:</strong> {company._location || '—'}</div>
            <div><strong>Last round:</strong> {company._last_round_amount_usd || '—'}</div>
            <div><strong>Investors:</strong> {company._num_investors || '—'}</div>
            <div><strong>Articles:</strong> {company._num_articles || '—'}</div>
            <div><strong>Days since last round:</strong> {company._days_since_last_round || '—'}</div>
          </div>
        </div>

        <div style={{background:'#15171c', padding:14, borderRadius:12, marginTop:12}}>
          <h4 style={{marginTop:0}}>Company metrics</h4>
          <Bar data={data} options={{plugins:{legend:{display:false}, tooltip:{enabled:true}}, scales:{y:{beginAtZero:true, ticks:{color:'#ddd'}}, x:{ticks:{color:'#ddd'}}}}} />
        </div>
      </div>
    </div>
  );
}
