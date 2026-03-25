import React from 'react'

const NAV_ITEMS = [
  { id: 'chat',       icon: '💬', label: 'Consulta Clínica' },
  { id: 'prontuario', icon: '📋', label: 'Prontuário Paciente' },
  { id: 'auditoria',  icon: '📊', label: 'Log e Auditoria' },
  { id: 'sobre',      icon: 'ℹ️', label: 'Sobre' },
]

export default function Sidebar({ page, setPage }) {
  return (
    <aside className="w-56 min-h-screen bg-slate-900 flex flex-col shrink-0 border-r border-slate-800">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-base shrink-0">
          ⚕
        </div>
        <span className="text-slate-100 font-semibold text-sm leading-tight">
          Assistente Médico
        </span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            onClick={() => setPage(item.id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left
              ${page === item.id
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'
              }`}
          >
            <span className="text-base leading-none">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-2 py-3 border-t border-slate-800">
        <div className="px-3 py-2 text-xs text-slate-500">
          FIAP PosTech · v1.0.0
        </div>
      </div>
    </aside>
  )
}
