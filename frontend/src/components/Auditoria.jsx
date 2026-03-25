import React, { useState, useEffect } from 'react'
import { api } from '../services/api.js'

export default function Auditoria() {
  const [logs, setLogs]       = useState([])
  const [stats, setStats]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)

  async function load() {
    setLoading(true)
    try {
      const [auditData, statsData] = await Promise.all([api.audit(20), api.stats()])
      setLogs(auditData.interacoes || auditData)
      setStats(statsData)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b border-slate-200 px-7 py-3.5 flex items-center justify-between shrink-0">
        <div>
          <span className="font-semibold text-slate-800">Log e Auditoria</span>
          <span className="text-slate-400 text-sm ml-2">Interações anonimizadas</span>
        </div>
        <button
          onClick={load}
          className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-600 px-3 py-1.5 rounded-lg transition-colors"
        >
          Atualizar
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-7 py-6">
        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            <StatCard label="Total de Interações" value={stats.total_interacoes ?? 0} color="blue" />
            <StatCard label="Alertas de Segurança" value={stats.total_alertas ?? 0} color="red" />
            <StatCard label="Taxa de Alertas" value={`${(stats.percentual_alertas ?? 0).toFixed(1)}%`} color="amber" />
          </div>
        )}

        {/* Table */}
        {loading ? (
          <p className="text-slate-400 text-sm">Carregando...</p>
        ) : logs.length === 0 ? (
          <p className="text-slate-400 text-sm">Nenhuma interação registrada ainda.</p>
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Timestamp</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Chain</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Alerta</th>
                  <th className="text-left px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Pergunta</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {logs.map((log, i) => (
                  <tr key={i} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3 text-slate-500 whitespace-nowrap text-xs">
                      {(log.timestamp || '').slice(0, 19).replace('T', ' ')}
                    </td>
                    <td className="px-4 py-3">
                      <span className="bg-blue-50 text-blue-700 text-xs px-2 py-0.5 rounded-full font-medium">
                        {log.chain_utilizada || '—'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {log.flag_alerta_seguranca
                        ? <span className="text-red-500">⚠️ Sim</span>
                        : <span className="text-green-600">✅ Não</span>}
                    </td>
                    <td className="px-4 py-3 text-slate-600 max-w-xs truncate text-xs">
                      {log.pergunta_anonimizada || '—'}
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => setSelected(log)}
                        className="text-xs text-blue-600 hover:underline"
                      >
                        Detalhes
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-6" onClick={() => setSelected(null)}>
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-xl max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-start justify-between mb-4">
              <h3 className="font-semibold text-slate-800">Detalhes da Interação</h3>
              <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-slate-600 text-xl leading-none">×</button>
            </div>
            <dl className="space-y-3 text-sm">
              <Row label="ID Interação"    value={selected.id_interacao} />
              <Row label="Sessão"          value={selected.id_sessao} />
              <Row label="Timestamp"       value={(selected.timestamp || '').slice(0, 19).replace('T', ' ')} />
              <Row label="Chain"           value={selected.chain_utilizada} />
              <Row label="Alerta"          value={selected.flag_alerta_seguranca ? '⚠️ Sim' : '✅ Não'} />
              <Row label="Confiança"       value={selected.confianca_estimada != null ? `${(selected.confianca_estimada * 100).toFixed(0)}%` : '—'} />
              <Row label="Motivo"          value={selected.motivo_resposta || '—'} />
              <Row label="Pergunta"        value={selected.pergunta_anonimizada} />
              <Row label="Resposta"        value={selected.resposta_assistente} multiline />
            </dl>
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color }) {
  const colors = {
    blue:  'bg-blue-50  text-blue-700  border-blue-100',
    red:   'bg-red-50   text-red-700   border-red-100',
    amber: 'bg-amber-50 text-amber-700 border-amber-100',
  }
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-xs mt-1 opacity-75">{label}</p>
    </div>
  )
}

function Row({ label, value, multiline }) {
  return (
    <div>
      <dt className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-0.5">{label}</dt>
      <dd className={`text-slate-700 ${multiline ? 'whitespace-pre-wrap' : ''}`}>{value || '—'}</dd>
    </div>
  )
}
