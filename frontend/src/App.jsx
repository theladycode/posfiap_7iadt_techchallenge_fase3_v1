import React, { useState } from 'react'
import Sidebar    from './components/Sidebar.jsx'
import Chat       from './components/Chat.jsx'
import Prontuario from './components/Prontuario.jsx'
import Auditoria  from './components/Auditoria.jsx'
import Sobre      from './components/Sobre.jsx'

const PAGES = {
  chat:       Chat,
  prontuario: Prontuario,
  auditoria:  Auditoria,
  sobre:      Sobre,
}

export default function App() {
  const [page, setPage] = useState('chat')
  const Page = PAGES[page] || Chat

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden">
      <Sidebar page={page} setPage={setPage} />
      <main className="flex-1 flex flex-col min-w-0 bg-slate-50">
        <Page />
      </main>
    </div>
  )
}
