const apiBase = process.env.REACT_APP_API_BASE || 'http://localhost:5001';

export async function fetchStatus() {
  const rsp = await fetch(`${apiBase}/api/status`);
  if (!rsp.ok) throw new Error('Failed to fetch status');
  return rsp.json();
}

export async function analyzeFile(file, onProgress) {
  const fd = new FormData();
  fd.append('file', file);
  const rsp = await fetch(`${apiBase}/api/analyze`, { method: 'POST', body: fd });
  if (!rsp.ok) {
    const t = await rsp.text();
    throw new Error(t || `Analyze failed (${rsp.status})`);
  }
  return rsp.json();
}

export function downloadUrl(ticket) {
  return `${apiBase}/api/download/${ticket}`;
}

export default { fetchStatus, analyzeFile, downloadUrl };
