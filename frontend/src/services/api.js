const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Erro na requisição')
  }
  return res.json()
}

export const api = {
  health: ()                => request('/health'),
  stats:  ()                => request('/stats'),
  audit:  (limit = 20)      => request(`/audit?limit=${limit}`),
  auditById: (id)           => request(`/audit/${id}`),

  chat: (pergunta)          => request('/chat',      { method: 'POST', body: JSON.stringify({ pergunta }) }),
  exams: (data)             => request('/exams',     { method: 'POST', body: JSON.stringify(data) }),
  treatment: (data)         => request('/treatment', { method: 'POST', body: JSON.stringify(data) }),
  alert: (data)             => request('/alert',     { method: 'POST', body: JSON.stringify(data) }),
  analyze: (data)           => request('/analyze',   { method: 'POST', body: JSON.stringify(data) }),
}
