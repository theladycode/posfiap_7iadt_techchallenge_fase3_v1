"""
app.py
------
Interface web do assistente médico usando Gradio.

Abas da interface:
1. Consulta Clínica     — Chat com o assistente médico via RAG + LLM
2. Análise de Paciente  — Inserir dados e rodar fluxo LangGraph completo
3. Logs e Auditoria     — Visualizar logs das interações recentes
4. Sobre               — Explicação do sistema e disclaimers

Tema: cores em azul médico com acessibilidade e clareza visual.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from assistant.chains import obter_assistente
from assistant.graph import processar_paciente
from assistant.logger import logger_auditoria

# ============================================================
# CONSTANTES
# ============================================================
COR_PRIMARIA = "#1a5276"
COR_SECUNDARIA = "#2e86c1"
COR_ALERTA = "#c0392b"
COR_SUCESSO = "#1e8449"
TITULO_APP = "Assistente Médico Virtual — IA de Apoio Clínico"
VERSAO = "1.0.0"

DISCLAIMER_PRINCIPAL = (
    "**AVISO IMPORTANTE:** Este sistema é um assistente de apoio à decisão clínica. "
    "As informações fornecidas são baseadas em protocolos e diretrizes médicas, mas "
    "**NÃO substituem a avaliação, diagnóstico ou prescrição de um profissional de saúde "
    "habilitado.** Em caso de emergência, ligue **192 (SAMU)** ou vá ao pronto-socorro."
)

# Configura o logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Inicializa o assistente de forma lazy
_assistente = None


def obter_assistente_lazy():
    """
    Inicializa o assistente médico apenas na primeira chamada (lazy loading).

    Retorna:
        Instância do assistente médico configurado
    """
    global _assistente
    if _assistente is None:
        logger.info("Inicializando assistente médico (primeira chamada)...")
        _assistente = obter_assistente()
    return _assistente


# ============================================================
# FUNÇÕES DE INTERFACE
# ============================================================

def processar_chat_clinico(mensagem: str, historico: list) -> tuple:
    """
    Processa mensagens do chat clínico e retorna resposta do assistente.

    Parâmetros:
        mensagem: Pergunta do usuário
        historico: Histórico de mensagens anteriores do chat

    Retorna:
        Tupla (string vazia para limpar input, historico atualizado)
    """
    if not mensagem.strip():
        return "", historico

    try:
        assistente = obter_assistente_lazy()
        resultado = assistente.responder_pergunta_clinica(mensagem)
        resposta = resultado.get("resposta", "Não foi possível gerar uma resposta.")

        # Adiciona ícone de emergência se necessário
        if resultado.get("e_emergencia"):
            resposta = "🚨 " + resposta

        # Adiciona fontes se disponíveis
        fontes = resultado.get("fontes", [])
        if fontes:
            fontes_unicas = list(set(fontes))[:3]
            resposta += f"\n\n*Fontes: {', '.join(fontes_unicas)}*"

    except Exception as erro:
        logger.error(f"Erro no chat clínico: {erro}")
        resposta = (
            f"Ocorreu um erro ao processar sua pergunta. "
            f"Por favor, tente novamente ou consulte um profissional de saúde."
        )

    historico.append({"role": "user", "content": mensagem})
    historico.append({"role": "assistant", "content": resposta})
    return "", historico


def limpar_chat_historico() -> tuple:
    """
    Limpa o histórico do chat e reinicia a memória conversacional.

    Retorna:
        Tupla com listas vazias para resetar o chat
    """
    try:
        assistente = obter_assistente_lazy()
        assistente.limpar_memoria()
    except Exception as erro:
        logger.warning(f"Erro ao limpar memória: {erro}")

    return [], []


def analisar_paciente_grafo(
    patient_id: str,
    sintomas: str,
    exames: str,
    historico: str,
    alergias: str,
    medicamentos: str
) -> tuple:
    """
    Executa o fluxo completo LangGraph para análise de um paciente.

    Mostra o progresso em tempo real e retorna o relatório final.

    Parâmetros:
        patient_id: ID do paciente (será anonimizado)
        sintomas: Sintomas e queixas atuais
        exames: Exames pendentes e resultados
        historico: Histórico clínico
        alergias: Alergias medicamentosas
        medicamentos: Medicamentos em uso

    Retorna:
        Tupla (log_progresso, relatorio_final, status_alertas)
    """
    if not sintomas.strip():
        return (
            "Por favor, informe pelo menos os sintomas do paciente.",
            "",
            "Nenhuma análise realizada."
        )

    try:
        log_progresso = "Iniciando análise do paciente...\n"
        log_progresso += f"ID: {patient_id or 'Não informado'}\n"
        log_progresso += f"Sintomas: {sintomas[:100]}...\n\n"
        log_progresso += "Executando fluxo LangGraph:\n"
        log_progresso += "  ✓ Etapa 1: Verificando exames pendentes...\n"
        log_progresso += "  ✓ Etapa 2: Consultando histórico clínico...\n"
        log_progresso += "  ✓ Etapa 3: Sugerindo conduta terapêutica...\n"
        log_progresso += "  ✓ Etapa 4: Validando segurança...\n"

        estado_final = processar_paciente(
            patient_id=patient_id or f"PAC-{datetime.now().strftime('%H%M%S')}",
            symptoms=sintomas,
            pending_exams=exames,
            history=historico,
            allergies=alergias or "Nenhuma conhecida",
            current_medications=medicamentos or "Nenhum"
        )

        alertas = estado_final.get("alerts", [])
        if alertas:
            log_progresso += f"  ⚠️  Etapa 5: {len(alertas)} alertas identificados — notificando equipe...\n"
        else:
            log_progresso += "  ✓ Etapa 5: Sem alertas críticos identificados.\n"

        log_progresso += "  ✓ Etapa 6: Gerando relatório final...\n\n"
        log_progresso += f"Etapas concluídas: {len(estado_final.get('etapas_executadas', []))}"

        relatorio = estado_final.get("final_response", "Relatório não disponível.")

        # Formata status dos alertas
        if alertas:
            status_alertas = f"⚠️ {len(alertas)} ALERTA(S) IDENTIFICADO(S):\n"
            for alerta in alertas:
                nivel = alerta.get("nivel", "INFO")
                descricao = alerta.get("descricao", "")
                status_alertas += f"• [{nivel}] {descricao}\n"
        else:
            status_alertas = "✅ Nenhum alerta crítico identificado na análise."

        return log_progresso, relatorio, status_alertas

    except Exception as erro:
        logger.error(f"Erro no processamento do grafo: {erro}")
        return (
            f"Erro durante a análise: {erro}",
            "Não foi possível gerar o relatório. Consulte o médico assistente.",
            "Erro no sistema."
        )


def carregar_logs_auditoria() -> tuple:
    """
    Carrega e formata os logs recentes para exibição na interface.

    Retorna:
        Tupla (dataframe_logs, texto_estatisticas)
    """
    try:
        interacoes = logger_auditoria.recuperar_interacoes_recentes(num_interacoes=20)
        estatisticas = logger_auditoria.obter_estatisticas()

        if not interacoes:
            df = pd.DataFrame(columns=[
                "Timestamp", "ID Sessão", "Chain", "Alerta", "Pergunta (resumo)"
            ])
        else:
            registros = []
            for interacao in interacoes:
                registros.append({
                    "Timestamp": interacao.get("timestamp", "")[:19].replace("T", " "),
                    "ID Sessão": interacao.get("id_sessao", ""),
                    "Chain": interacao.get("chain_utilizada", ""),
                    "Alerta": "⚠️ Sim" if interacao.get("flag_alerta_seguranca") else "✅ Não",
                    "Pergunta (resumo)": interacao.get("pergunta_anonimizada", "")[:80] + "..."
                })
            df = pd.DataFrame(registros)

        texto_estat = (
            f"**Total de interações:** {estatisticas.get('total_interacoes', 0)}\n"
            f"**Total de alertas:** {estatisticas.get('total_alertas', 0)}\n"
            f"**Taxa de alertas:** {estatisticas.get('percentual_alertas', 0):.1f}%"
        )

        return df, texto_estat

    except Exception as erro:
        logger.error(f"Erro ao carregar logs: {erro}")
        df = pd.DataFrame({"Erro": [str(erro)]})
        return df, "Erro ao carregar estatísticas."


# ============================================================
# CONSTRUÇÃO DA INTERFACE GRADIO
# ============================================================

def construir_interface() -> gr.Blocks:
    """
    Constrói e retorna a interface Gradio completa do assistente médico.

    Retorna:
        Objeto gr.Blocks com todas as abas configuradas
    """
    css = """
            .disclaimer-box {
                background-color: #fef9c3;
                border-left: 4px solid #ca8a04;
                padding: 12px 16px;
                border-radius: 4px;
                margin-bottom: 16px;
            }
            .alerta-critico {
                background-color: #fee2e2;
                border-left: 4px solid #dc2626;
                padding: 12px 16px;
                border-radius: 4px;
            }
            .header-medico {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #1a5276, #2e86c1);
                color: white;
                border-radius: 8px;
                margin-bottom: 16px;
            }
        """

    with gr.Blocks(title=TITULO_APP, css=css) as interface:

        # Cabeçalho principal
        gr.HTML(f"""
            <div class="header-medico">
                <h1>Assistente Médico Virtual</h1>
                <p>Sistema de Apoio à Decisão Clínica | LLaMA 3 + LangChain + LangGraph</p>
                <small>Versão {VERSAO} | FIAP PosTech — Pós-graduação em IA</small>
            </div>
        """)

        # Disclaimer obrigatório no topo
        gr.HTML(f"""
            <div class="disclaimer-box">
                <strong>AVISO IMPORTANTE:</strong> Este sistema é um assistente de apoio à
                decisão clínica. As informações são baseadas em protocolos médicos e
                <strong>NÃO substituem</strong> a avaliação de um profissional de saúde habilitado.
                Em emergências: <strong>SAMU 192</strong>.
            </div>
        """)

        with gr.Tabs():

            # ================================================
            # ABA 1: CONSULTA CLÍNICA
            # ================================================
            with gr.Tab("💬 Consulta Clínica"):
                gr.Markdown(
                    "### Chat com o Assistente Médico\n"
                    "Faça perguntas clínicas e obtenha respostas baseadas em protocolos e diretrizes médicas. "
                    "O assistente mantém o contexto da conversa para perguntas de acompanhamento."
                )

                chatbot = gr.Chatbot(
                    label="Conversa com o Assistente Médico",
                    height=500,
                    show_label=True,
                )

                with gr.Row():
                    campo_mensagem = gr.Textbox(
                        placeholder="Digite sua pergunta clínica aqui... (ex: Qual o tratamento de primeira linha para HAS estágio 1?)",
                        label="Sua pergunta",
                        lines=2,
                        scale=5,
                    )

                with gr.Row():
                    btn_enviar = gr.Button("Enviar Pergunta", variant="primary", scale=2)
                    btn_limpar = gr.Button("Nova Conversa", variant="secondary", scale=1)

                # Exemplos de perguntas
                gr.Examples(
                    examples=[
                        "Qual o tratamento de primeira linha para hipertensão arterial estágio 1?",
                        "Como diagnosticar diabetes mellitus tipo 2?",
                        "Quais são os critérios diagnósticos para sepse?",
                        "Como manejar uma crise asmática grave no pronto-socorro?",
                        "Quais são os sinais de alarme na cefaleia que indicam investigação urgente?",
                    ],
                    inputs=campo_mensagem,
                    label="Exemplos de Perguntas Clínicas"
                )

                # Eventos do chat
                btn_enviar.click(
                    processar_chat_clinico,
                    inputs=[campo_mensagem, chatbot],
                    outputs=[campo_mensagem, chatbot]
                )
                campo_mensagem.submit(
                    processar_chat_clinico,
                    inputs=[campo_mensagem, chatbot],
                    outputs=[campo_mensagem, chatbot]
                )
                btn_limpar.click(
                    limpar_chat_historico,
                    inputs=[],
                    outputs=[chatbot, chatbot]
                )

            # ================================================
            # ABA 2: ANÁLISE DE PACIENTE (LANGGRAPH)
            # ================================================
            with gr.Tab("📋 Análise de Paciente"):
                gr.Markdown(
                    "### Análise Completa via Fluxo LangGraph\n"
                    "Insira os dados do paciente para executar o fluxo automatizado de análise clínica. "
                    "O sistema percorrerá todas as etapas do grafo e gerará um relatório completo."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Dados do Paciente")
                        campo_patient_id = gr.Textbox(
                            label="ID do Paciente (será anonimizado)",
                            placeholder="Ex: PAC-2024-001",
                        )
                        campo_sintomas = gr.Textbox(
                            label="Sintomas e Queixas Atuais *",
                            placeholder="Descreva os sintomas, há quanto tempo, intensidade...",
                            lines=4,
                        )
                        campo_exames = gr.Textbox(
                            label="Exames Disponíveis / Resultados",
                            placeholder="Ex: Hemograma: Hb 9g/dL, Leucócitos 15.000, PCR 120mg/L...",
                            lines=4,
                        )
                        campo_historico = gr.Textbox(
                            label="Histórico Clínico Relevante",
                            placeholder="Comorbidades, cirurgias prévias, internações...",
                            lines=3,
                        )
                        campo_alergias = gr.Textbox(
                            label="Alergias Medicamentosas",
                            placeholder="Ex: Alergia a penicilina, AINE...",
                        )
                        campo_medicamentos = gr.Textbox(
                            label="Medicamentos em Uso",
                            placeholder="Ex: Metformina 850mg 2x/dia, Losartana 50mg...",
                            lines=2,
                        )
                        btn_analisar = gr.Button(
                            "Executar Análise LangGraph",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Resultado da Análise")
                        saida_progresso = gr.Textbox(
                            label="Progresso do Fluxo LangGraph",
                            lines=10,
                            interactive=False,
                        )
                        saida_alertas = gr.Textbox(
                            label="Status de Alertas",
                            lines=4,
                            interactive=False,
                        )
                        saida_relatorio = gr.Textbox(
                            label="Relatório Clínico Final",
                            lines=20,
                            interactive=False,
                        )

                btn_analisar.click(
                    analisar_paciente_grafo,
                    inputs=[
                        campo_patient_id,
                        campo_sintomas,
                        campo_exames,
                        campo_historico,
                        campo_alergias,
                        campo_medicamentos,
                    ],
                    outputs=[saida_progresso, saida_relatorio, saida_alertas]
                )

            # ================================================
            # ABA 3: LOGS E AUDITORIA
            # ================================================
            with gr.Tab("📊 Logs e Auditoria"):
                gr.Markdown(
                    "### Registro de Interações\n"
                    "Visualize as interações recentes com o sistema. "
                    "Todos os dados são anonimizados automaticamente antes do registro."
                )

                with gr.Row():
                    btn_atualizar_logs = gr.Button("Atualizar Logs", variant="secondary")

                tabela_logs = gr.Dataframe(
                    label="Últimas 20 interações (dados anonimizados)",
                    interactive=False,
                    wrap=True,
                )

                texto_estatisticas = gr.Markdown(label="Estatísticas de Uso")

                btn_atualizar_logs.click(
                    carregar_logs_auditoria,
                    inputs=[],
                    outputs=[tabela_logs, texto_estatisticas]
                )

                # Carrega logs inicialmente
                interface.load(
                    carregar_logs_auditoria,
                    inputs=[],
                    outputs=[tabela_logs, texto_estatisticas]
                )

            # ================================================
            # ABA 4: SOBRE
            # ================================================
            with gr.Tab("ℹ️ Sobre"):
                gr.Markdown(f"""
# Sobre o Assistente Médico Virtual

**Versão:** {VERSAO} | **Instituição:** FIAP PosTech

---

## O que é este sistema?

Este é um **sistema de apoio à decisão clínica** desenvolvido como projeto acadêmico
de pós-graduação em Inteligência Artificial. Utiliza tecnologias de ponta em IA
generativa para auxiliar profissionais de saúde.

## Arquitetura Tecnológica

| Componente | Tecnologia |
|---|---|
| **Modelo de Linguagem** | LLaMA 3 8B (fine-tunado com QLoRA) |
| **Orquestração** | LangChain + LangGraph |
| **Recuperação de Contexto** | RAG com FAISS + SentenceTransformers |
| **Interface** | Gradio |
| **Infraestrutura** | Docker + Docker Compose |

## Fluxo LangGraph

```
[Entrada do Paciente]
        ↓
[Verificar Exames Pendentes]
        ↓
[Consultar Histórico Clínico]
        ↓
[Sugerir Conduta / Tratamento]
        ↓
[Validação de Segurança]
        ↓ ← → [Notificar Equipe Médica] (se alertas)
[Resposta Final ao Médico]
```

## Módulos do Sistema

- **prepare_dataset.py** — Processamento e anonimização de dados médicos
- **train.py** — Fine-tuning com QLoRA (4-bit quantization)
- **evaluate.py** — Avaliação com métricas ROUGE
- **chains.py** — Chains LangChain para diferentes tarefas clínicas
- **graph.py** — Fluxo automatizado LangGraph
- **safety.py** — Validação e limites de atuação
- **logger.py** — Auditoria de todas as interações

## Limitações e Disclaimers

⚠️ **Este sistema:**
- É um **protótipo acadêmico** em desenvolvimento
- **NÃO** é certificado para uso clínico real
- **NÃO** substitui avaliação médica profissional
- **NÃO** deve ser usado como base única para diagnóstico ou tratamento
- Pode conter **imprecisões** nas informações geradas

## Segurança e Privacidade

- Todos os dados são anonimizados antes do registro em logs
- Nenhuma informação pessoal é armazenada permanentemente
- O sistema não transmite dados para servidores externos

---

*Desenvolvido para fins acadêmicos — FIAP PosTech, 2024*
                """)

    return interface


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    porta = int(os.environ.get("APP_PORT", 7860))
    compartilhar = os.environ.get("GRADIO_SHARE", "false").lower() == "true"

    # Pré-carrega o modelo na inicialização do container (antes de aceitar requests)
    logger.info("Pré-carregando modelo LLM na inicialização...")
    obter_assistente_lazy()
    logger.info("Modelo pronto. Iniciando interface Gradio...")

    logger.info(f"Iniciando interface Gradio na porta {porta}...")

    interface = construir_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=porta,
        share=compartilhar,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
    )
