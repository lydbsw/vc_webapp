import React from 'react';
import './PreviewTable.css';

export default function PreviewTable({ columns = [], rows = [], selectedCompany, onSelect }) {
  return (
    <div className="preview-root">
      <div className="preview-table-wrap">
        <table className="preview-table">
          <thead>
            <tr>
              {columns.map(c=> <th key={c} className="preview-th">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r,idx)=>(
              <tr
                key={idx}
                onClick={() => onSelect(r)}
                className={`preview-row ${selectedCompany && selectedCompany._key_name === r._key_name ? 'selected' : ''}`}
              >
                {columns.map(c=> <td key={c} className="preview-td">{r[c] ?? ''}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
