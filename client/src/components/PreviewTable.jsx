import React from 'react';

export default function PreviewTable({ columns = [], rows = [], selectedCompany, onSelect }) {
  return (
    <div style={{flex:1, minWidth:0, background:'#0b0c10', padding:0, maxHeight:'60vh', overflowY:'auto'}}>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%', borderCollapse:'collapse'}}>
          <thead>
            <tr>
              {columns.map(c=> <th key={c} style={{textAlign:'left', borderBottom:'1px solid #222', padding:8}}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r,idx)=>(
              <tr key={idx}
                  onClick={() => onSelect(r)}
                  style={{cursor:'pointer', background: selectedCompany && selectedCompany._key_name === r._key_name ? '#0f1720' : 'transparent'}}>
                {columns.map(c=> <td key={c} style={{borderBottom:'1px solid #222', padding:8}}>{r[c] ?? ''}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
