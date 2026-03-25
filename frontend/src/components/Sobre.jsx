import React from 'react'

const TECH = [
  ['Modelo de Linguagem', 'LLaMA 3 8B / Mistral-7B (QLoRA fine-tuning)'],
  ['Orquestração',        'LangChain + LangGraph'],
  ['Recuperação RAG',     'FAISS + SentenceTransformers'],
  ['API REST',            'FastAPI + Swagger UI'],
  ['Interface',           'React + Vite + Tailwind CSS'],
  ['Infraestrutura',      'Docker + Docker Compose'],
]

export default function Sobre() {
  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b border-slate-200 px-7 py-3.5 shrink-0">
        <span className="font-semibold text-slate-800">Sobre o Sistema</span>
      </div>

      <div className="flex-1 overflow-y-auto px-7 py-8 max-w-3xl">
        <div className="flex items-center gap-4 mb-8">
          <div className="w-14 h-14 rounded-2xl bg-blue-600 flex items-center justify-center text-2xl shrink-0">⚕</div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">Assistente Médico Virtual</h1>
            <p className="text-slate-500 text-sm mt-0.5">FIAP PosTech · Pós-graduação em IA · v1.0.0</p>
          </div>
        </div>

        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">O que é este sistema?</h2>
          <p className="text-slate-700 text-sm leading-relaxed">
            Sistema de <strong>apoio à decisão clínica</strong> desenvolvido como projeto acadêmico de
            pós-graduação em Inteligência Artificial. Utiliza LLMs fine-tunados com dados médicos,
            recuperação aumentada (RAG) e fluxos automatizados (LangGraph) para auxiliar profissionais de saúde.
          </p>
        </section>

        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Arquitetura Tecnológica</h2>
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            {TECH.map(([comp, tech], i) => (
              <div key={comp} className={`flex px-4 py-3 text-sm ${i % 2 === 0 ? 'bg-slate-50' : ''}`}>
                <span className="font-medium text-slate-700 w-48 shrink-0">{comp}</span>
                <span className="text-slate-500">{tech}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="mb-8">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Fluxo LangGraph</h2>
          <div className="bg-slate-900 rounded-xl p-4 text-xs text-slate-300 font-mono leading-loose">
            {`[Entrada do Paciente]
      ↓
[Verificar Exames Pendentes]
      ↓
[Consultar Histórico Clínico]
      ↓
[Sugerir Conduta / Tratamento]
      ↓
[Validação de Segurança]
      ↓ ←→ [Notificar Equipe] (se alertas)
[Resposta Final ao Médico]`}
          </div>
        </section>

        <section>
          <div className="bg-amber-50 border border-amber-200 border-l-4 border-l-amber-400 rounded-xl p-4 text-sm text-amber-800">
            <p className="font-semibold mb-1">⚠️ Limitações Importantes</p>
            <ul className="space-y-1 text-xs list-disc list-inside">
              <li>Este é um <strong>protótipo acadêmico</strong> e não é certificado para uso clínico real</li>
              <li><strong>NÃO</strong> substitui avaliação, diagnóstico ou prescrição médica profissional</li>
              <li>Pode conter imprecisões nas informações geradas</li>
              <li>Em caso de emergência: ligue <strong>SAMU 192</strong></li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  )
}
