import React, { useState } from 'react'
import { api } from '../services/api.js'

const EMPTY = { patient_id: '', sintomas: '', exames: '', historico: '', alergias: '', medicamentos: '' }

export default function Prontuario() {
  const [form, setForm]       = useState(EMPTY)
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  function handleChange(e) {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.sintomas.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.analyze({
        patient_id:         form.patient_id   || undefined,
        sintomas:           form.sintomas,
        exames:             form.exames       || undefined,
        historico:          form.historico    || undefined,
        alergias:           form.alergias     || undefined,
        medicamentos_em_uso: form.medicamentos || undefined,
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b border-slate-200 px-7 py-3.5 shrink-0">
        <span className="font-semibold text-slate-800">Prontuário do Paciente</span>
        <span className="text-slate-400 text-sm ml-2">Análise via LangGraph</span>
      </div>

      <div className="flex-1 overflow-y-auto px-7 py-6">
        <div className="grid grid-cols-2 gap-6 max-w-5xl">

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <h3 className="font-semibold text-slate-700 text-sm uppercase tracking-wide">
              Dados do Paciente
            </h3>

            <Field label="ID do Paciente" name="patient_id" placeholder="Ex: PAC-2024-001" value={form.patient_id} onChange={handleChange} />
            <Field label="Sintomas e Queixas *" name="sintomas" placeholder="Descreva os sintomas, duração, intensidade..." value={form.sintomas} onChange={handleChange} rows={4} required />
            <Field label="Exames Disponíveis / Resultados" name="exames" placeholder="Ex: Hb 9g/dL, Leucócitos 15.000..." value={form.exames} onChange={handleChange} rows={3} />
            <Field label="Histórico Clínico" name="historico" placeholder="Comorbidades, cirurgias prévias..." value={form.historico} onChange={handleChange} rows={3} />
            <Field label="Alergias Medicamentosas" name="alergias" placeholder="Ex: Penicilina, AINE..." value={form.alergias} onChange={handleChange} />
            <Field label="Medicamentos em Uso" name="medicamentos" placeholder="Ex: Metformina 850mg 2x/dia..." value={form.medicamentos} onChange={handleChange} rows={2} />

            <button
              type="submit"
              disabled={loading || !form.sintomas.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 rounded-xl
                text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analisando...' : 'Executar Análise LangGraph'}
            </button>
          </form>

          {/* Results */}
          <div className="space-y-4">
            <h3 className="font-semibold text-slate-700 text-sm uppercase tracking-wide">
              Resultado da Análise
            </h3>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl p-4">
                {error}
              </div>
            )}

            {!result && !error && !loading && (
              <p className="text-slate-400 text-sm">Preencha os dados e clique em analisar.</p>
            )}

            {loading && (
              <div className="space-y-3">
                {['Verificando exames...', 'Consultando histórico...', 'Sugerindo conduta...', 'Validando segurança...'].map(step => (
                  <div key={step} className="flex items-center gap-2 text-sm text-slate-500">
                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    {step}
                  </div>
                ))}
              </div>
            )}

            {result && (
              <>
                {result.alertas && result.alertas.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                    <p className="font-semibold text-red-700 text-sm mb-2">⚠️ {result.alertas.length} Alerta(s)</p>
                    {result.alertas.map((a, i) => (
                      <p key={i} className="text-red-600 text-xs">• [{a.nivel}] {a.descricao}</p>
                    ))}
                  </div>
                )}
                <div className="bg-white border border-slate-200 rounded-xl p-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Relatório Final</p>
                  <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">{result.relatorio || result.resposta}</p>
                  {result.id_interacao && (
                    <p className="mt-3 text-xs text-slate-400">ID: {result.id_interacao}</p>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, name, value, onChange, placeholder, rows, required }) {
  const cls = "w-full border border-slate-200 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
  return (
    <div>
      <label className="block text-xs font-medium text-slate-600 mb-1">{label}</label>
      {rows
        ? <textarea name={name} value={value} onChange={onChange} placeholder={placeholder} rows={rows} required={required} className={cls} />
        : <input   name={name} value={value} onChange={onChange} placeholder={placeholder} required={required} className={cls} />
      }
    </div>
  )
}
