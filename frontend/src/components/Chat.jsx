import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api.js'

const SUGGESTIONS = [
  'Tratamento de primeira linha para hipertensão?',
  'Quais são os critérios diagnósticos para sepse?',
  'Como manejar uma crise asmática grave no PR?',
]

function LoadingDots() {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      {[0,1,2].map(i => (
        <span key={i} className="dot w-2 h-2 rounded-full bg-slate-400 block" />
      ))}
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`msg-enter flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs mr-2 mt-0.5 shrink-0">
          ⚕
        </div>
      )}
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed
          ${isUser
            ? 'bg-blue-600 text-white rounded-br-sm'
            : 'bg-white text-slate-800 shadow-sm border border-slate-100 rounded-bl-sm'
          }`}
      >
        {isUser ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
        {msg.id_interacao && (
          <div className="mt-1.5 text-xs text-slate-400">
            ID: {msg.id_interacao}
          </div>
        )}
      </div>
    </div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(text) {
    const pergunta = (text || input).trim()
    if (!pergunta || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: pergunta }])
    setLoading(true)

    try {
      const data = await api.chat(pergunta)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.resposta,
        id_interacao: data.id_interacao,
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Erro: ${err.message}`,
      }])
    } finally {
      setLoading(false)
    }
  }

  function clear() {
    setMessages([])
    setInput('')
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-7 py-3.5 flex items-center gap-2 text-sm text-slate-500 shrink-0">
        <span className="font-semibold text-slate-800">Assistente Médico</span>
        <span className="text-slate-300">|</span>
        <span>LLaMA-3 · LangChain</span>
      </div>

      {/* Greeting */}
      <div className="px-7 pt-7 pb-2 shrink-0">
        <h2 className="text-2xl font-bold text-slate-900">Olá, Dr. João!</h2>
        <p className="text-slate-500 mt-1 text-sm">Como posso ajudar você hoje?</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-7 py-4">
        {messages.length === 0 && (
          <p className="text-center text-slate-400 text-sm mt-8">
            Faça sua pergunta clínica abaixo para começar.
          </p>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && (
          <div className="flex justify-start mb-3">
            <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs mr-2 mt-0.5 shrink-0">⚕</div>
            <div className="bg-white rounded-2xl rounded-bl-sm shadow-sm border border-slate-100">
              <LoadingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Warning */}
      <div className="px-7 pb-2 shrink-0">
        <div className="bg-amber-50 border border-amber-200 border-l-4 border-l-amber-400 rounded-lg px-4 py-2.5 text-xs text-amber-800 leading-relaxed">
          <strong>⚠️ Aviso Importante:</strong> Este sistema é de apoio à decisão clínica e{' '}
          <strong>NÃO</strong> substitui avaliação, diagnóstico ou prescrição médica.
          Em emergências: <strong>SAMU 192</strong>.
        </div>
      </div>

      {/* Suggestions */}
      <div className="px-7 pb-2 flex gap-2 flex-wrap shrink-0">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => send(s)}
            disabled={loading}
            className="text-xs bg-white border border-slate-200 text-slate-600 rounded-full px-3 py-1.5
              hover:bg-slate-50 hover:border-slate-300 transition-colors disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 px-7 py-4 flex gap-3 shrink-0">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Digite sua pergunta clínica aqui..."
          disabled={loading}
          className="flex-1 border border-slate-200 rounded-xl px-4 py-2.5 text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
            disabled:bg-slate-50 disabled:text-slate-400"
        />
        <button
          onClick={() => send()}
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-5 py-2.5 rounded-xl
            transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Enviar
        </button>
        <button
          onClick={clear}
          disabled={loading}
          className="bg-slate-100 hover:bg-slate-200 text-slate-600 text-sm font-medium px-4 py-2.5 rounded-xl
            transition-colors disabled:opacity-50"
        >
          Nova Conversa
        </button>
      </div>
    </div>
  )
}
