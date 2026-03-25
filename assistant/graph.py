"""
graph.py
--------
Define o fluxo automatizado de decisão clínica usando LangGraph.

Fluxo principal ao receber dados de um paciente:
  [Entrada do paciente]
         ↓
  [Verificar exames pendentes]
         ↓
  [Consultar histórico clínico]
         ↓
  [Sugerir conduta / tratamento]
         ↓
  [Validação de segurança]
         ↓
  [Emitir alerta se necessário]
         ↓
  [Resposta final ao médico]

O grafo é implementado com StateGraph do LangGraph, onde cada nó
é uma função pura que recebe e retorna o estado do paciente.
"""

import logging
from typing import Optional, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from assistant.chains import obter_assistente
from assistant.safety import validador_seguranca
from assistant.logger import logger_auditoria

# Configura o logger
logger = logging.getLogger(__name__)


# ============================================================
# ESTADO DO GRAFO
# ============================================================

class PatientState(TypedDict):
    """
    Estado compartilhado entre todos os nós do grafo LangGraph.

    Cada nó lê o estado atual, executa sua lógica e retorna
    um dicionário com os campos atualizados.

    Atributos:
        patient_id: Identificador do paciente (anonimizado)
        symptoms: Sintomas e queixas relatadas
        pending_exams: Lista de exames pendentes e resultados
        history: Histórico clínico relevante
        allergies: Alergias medicamentosas conhecidas
        current_medications: Medicamentos em uso
        suggested_treatment: Sugestão de conduta terapêutica
        alerts: Lista de alertas clínicos identificados
        exam_analysis: Análise dos exames laboratoriais
        safety_validated: Flag indicando se passou pela validação de segurança
        final_response: Resposta consolidada para o médico
        etapas_executadas: Registro das etapas percorridas no grafo
    """
    patient_id: str
    symptoms: str
    pending_exams: str
    history: str
    allergies: str
    current_medications: str
    suggested_treatment: str
    alerts: list
    exam_analysis: str
    safety_validated: bool
    final_response: str
    etapas_executadas: list


# ============================================================
# NÓS DO GRAFO
# ============================================================

def verificar_exames_pendentes(state: PatientState) -> PatientState:
    """
    Nó 1: Verifica e analisa os exames pendentes do paciente.

    Consulta o assistente para interpretar resultados laboratoriais
    e identificar achados relevantes para o diagnóstico.

    Parâmetros:
        state: Estado atual do paciente

    Retorna:
        Estado atualizado com a análise dos exames
    """
    logger.info(f"[Grafo] Nó 1: Analisando exames do paciente {state['patient_id']}")

    assistente = obter_assistente()
    exames = state.get("pending_exams", "Nenhum exame informado")
    historico = state.get("history", "Não disponível")
    sintomas = state.get("symptoms", "Não informado")

    if exames and exames.strip() and exames != "Nenhum exame informado":
        resultado = assistente.analisar_exames(
            exames=exames,
            historico=historico,
            queixas=sintomas
        )
        analise = resultado.get("resposta", "Não foi possível analisar os exames.")
    else:
        analise = "Não há exames pendentes para análise neste momento."

    etapas = state.get("etapas_executadas", [])
    etapas.append("verificar_exames_pendentes")

    return {
        **state,
        "exam_analysis": analise,
        "etapas_executadas": etapas,
    }


def consultar_historico_clinico(state: PatientState) -> PatientState:
    """
    Nó 2: Consulta e sintetiza o histórico clínico do paciente.

    Analisa o histórico em conjunto com os sintomas atuais para
    identificar padrões clínicos relevantes e comorbidades.

    Parâmetros:
        state: Estado atual do paciente

    Retorna:
        Estado atualizado após análise do histórico
    """
    logger.info(f"[Grafo] Nó 2: Consultando histórico clínico de {state['patient_id']}")

    assistente = obter_assistente()
    historico = state.get("history", "")
    sintomas = state.get("symptoms", "")

    # Constrói a pergunta de análise do histórico
    if historico and historico.strip():
        pergunta_historico = (
            f"Analise o seguinte histórico clínico e sua relação com os sintomas atuais:\n"
            f"Histórico: {historico}\n"
            f"Sintomas atuais: {sintomas}\n"
            f"Identifique comorbidades relevantes e fatores de risco."
        )
        resultado = assistente.responder_pergunta_clinica(pergunta_historico)
        analise_historico = resultado.get("resposta", "Histórico analisado.")
    else:
        analise_historico = "Histórico clínico não disponível. Avaliação baseada nos sintomas atuais."

    etapas = state.get("etapas_executadas", [])
    etapas.append("consultar_historico_clinico")

    return {
        **state,
        "history": f"{historico}\n\n[Síntese do Assistente]: {analise_historico}",
        "etapas_executadas": etapas,
    }


def sugerir_conduta_tratamento(state: PatientState) -> PatientState:
    """
    Nó 3: Sugere conduta terapêutica baseada nos dados disponíveis.

    Integra histórico, sintomas e análise de exames para propor
    tratamento baseado em diretrizes e protocolos clínicos.

    Parâmetros:
        state: Estado atual do paciente

    Retorna:
        Estado atualizado com sugestão terapêutica
    """
    logger.info(f"[Grafo] Nó 3: Gerando sugestão terapêutica para {state['patient_id']}")

    assistente = obter_assistente()
    sintomas = state.get("symptoms", "Não informado")
    historico = state.get("history", "Não disponível")
    alergias = state.get("allergies", "Nenhuma conhecida")
    medicamentos = state.get("current_medications", "Nenhum")
    analise_exames = state.get("exam_analysis", "")

    # Constrói hipótese diagnóstica com base nos dados disponíveis
    hipotese = (
        f"Paciente com: {sintomas}. "
        f"Exames: {analise_exames[:200] if analise_exames else 'Sem exames disponíveis'}"
    )

    resultado = assistente.sugerir_tratamento(
        diagnostico_hipotetico=hipotese,
        historico=historico,
        alergias=alergias,
        medicamentos_em_uso=medicamentos
    )

    sugestao = resultado.get("resposta", "Não foi possível gerar sugestão terapêutica.")

    etapas = state.get("etapas_executadas", [])
    etapas.append("sugerir_conduta_tratamento")

    return {
        **state,
        "suggested_treatment": sugestao,
        "etapas_executadas": etapas,
    }


def validar_seguranca(state: PatientState) -> PatientState:
    """
    Nó 4: Valida todo o conteúdo gerado pelas etapas anteriores.

    Aplica as regras de segurança do módulo safety.py para garantir
    que nenhuma resposta ultrapasse os limites éticos do sistema.

    Parâmetros:
        state: Estado atual do paciente

    Retorna:
        Estado atualizado com flag de validação e alertas identificados
    """
    logger.info(f"[Grafo] Nó 4: Validando segurança do conteúdo gerado")

    alertas_atuais = list(state.get("alerts", []))
    sugestao = state.get("suggested_treatment", "")
    sintomas = state.get("symptoms", "")
    analise = state.get("exam_analysis", "")

    # Valida a sugestão de tratamento
    if sugestao:
        validacao = validador_seguranca.validar_resposta(
            pergunta=sintomas,
            resposta=sugestao,
            chain_utilizada="graph_treatment"
        )

        if validacao.e_emergencia:
            alertas_atuais.append({
                "tipo": "EMERGENCIA",
                "descricao": "Padrão de emergência detectado nos dados do paciente",
                "acao": "Notificar equipe médica imediatamente",
                "nivel": "CRÍTICO"
            })

        if validacao.flag_alerta:
            alertas_atuais.append({
                "tipo": "SEGURANCA",
                "descricao": "Módulo de segurança interveio na resposta",
                "intervencoes": validacao.intervencoes,
                "nivel": "ALTO"
            })

    # Verifica achados críticos nos exames
    achados_criticos = _identificar_achados_criticos(analise)
    if achados_criticos:
        alertas_atuais.append({
            "tipo": "ACHADO_CRITICO",
            "descricao": achados_criticos,
            "acao": "Revisar exames e acionar especialista",
            "nivel": "ALTO"
        })

    etapas = state.get("etapas_executadas", [])
    etapas.append("validar_seguranca")

    return {
        **state,
        "alerts": alertas_atuais,
        "safety_validated": True,
        "etapas_executadas": etapas,
    }


def _identificar_achados_criticos(texto_exames: str) -> Optional[str]:
    """
    Identifica termos críticos nos resultados de exames.

    Parâmetros:
        texto_exames: Texto com análise dos exames

    Retorna:
        Descrição dos achados críticos ou None se não houver
    """
    if not texto_exames:
        return None

    termos_criticos = [
        "hiperkalemia grave", "hipercalemia", "k+ > 6",
        "acidose grave", "ph < 7.1", "lactato > 4",
        "troponina elevada", "st elevado", "choque",
        "insuficiência renal aguda", "creatinina > 5",
        "sangramento", "hemoglobina < 7", "plaquetas < 50",
    ]

    texto_lower = texto_exames.lower()
    achados = [termo for termo in termos_criticos if termo in texto_lower]

    if achados:
        return f"Possíveis achados críticos identificados: {', '.join(achados)}"

    return None


def notificar_equipe_medica(state: PatientState) -> PatientState:
    """
    Nó 5 (condicional): Notifica a equipe médica sobre alertas críticos.

    Este nó é ativado apenas quando existem alertas pendentes no estado.
    Gera um alerta estruturado e registra na auditoria com prioridade.

    Parâmetros:
        state: Estado atual do paciente com alertas identificados

    Retorna:
        Estado atualizado após emissão dos alertas
    """
    logger.warning(
        f"[Grafo] Nó 5 (ALERTA): Notificando equipe médica sobre "
        f"{len(state.get('alerts', []))} alertas para paciente {state['patient_id']}"
    )

    assistente = obter_assistente()
    alertas = state.get("alerts", [])

    # Formata os alertas para geração da notificação
    alertas_texto = "\n".join([
        f"• [{a.get('nivel', 'INFO')}] {a.get('tipo', 'ALERTA')}: {a.get('descricao', '')}"
        for a in alertas
    ])

    dados_paciente = (
        f"ID: {state['patient_id']} | "
        f"Sintomas: {state.get('symptoms', 'N/A')[:100]}"
    )

    resultado_alerta = assistente.gerar_alerta(
        dados_paciente=dados_paciente,
        achados_criticos=alertas_texto
    )

    # Registra o alerta na auditoria com prioridade máxima
    logger_auditoria.registrar_interacao(
        pergunta_usuario=f"[ALERTA AUTOMATICO] Paciente: {state['patient_id']}",
        resposta_assistente=resultado_alerta.get("resposta", ""),
        chain_utilizada="graph_alert",
        flag_alerta=True,
        metadata_adicional={"alertas": alertas, "etapa": "notificacao"}
    )

    etapas = state.get("etapas_executadas", [])
    etapas.append("notificar_equipe_medica")

    return {
        **state,
        "etapas_executadas": etapas,
    }


def gerar_resposta_final(state: PatientState) -> PatientState:
    """
    Nó 6: Consolida todas as informações em uma resposta final estruturada.

    Compila análise de exames, histórico, sugestão terapêutica e alertas
    em um relatório final para o médico assistente.

    Parâmetros:
        state: Estado completo do paciente após todas as etapas

    Retorna:
        Estado final com resposta consolidada
    """
    logger.info(f"[Grafo] Nó 6: Gerando resposta final para paciente {state['patient_id']}")

    secoes = []
    secoes.append(f"# Relatório de Apoio Clínico — Paciente: {state['patient_id']}")
    secoes.append(f"## Etapas executadas: {' → '.join(state.get('etapas_executadas', []))}")
    secoes.append("")

    # Seção 1: Sintomas
    if state.get("symptoms"):
        secoes.append("## Queixas e Sintomas")
        secoes.append(state["symptoms"])
        secoes.append("")

    # Seção 2: Análise de exames
    if state.get("exam_analysis"):
        secoes.append("## Análise de Exames")
        secoes.append(state["exam_analysis"])
        secoes.append("")

    # Seção 3: Sugestão terapêutica
    if state.get("suggested_treatment"):
        secoes.append("## Sugestão de Conduta")
        secoes.append(state["suggested_treatment"])
        secoes.append("")

    # Seção 4: Alertas (se houver)
    alertas = state.get("alerts", [])
    if alertas:
        secoes.append("## ⚠️ Alertas Identificados")
        for alerta in alertas:
            secoes.append(
                f"- **[{alerta.get('nivel', 'INFO')}]** {alerta.get('descricao', '')}"
            )
        secoes.append("")

    # Rodapé obrigatório
    secoes.append(
        "---\n*Este relatório foi gerado por sistema de apoio à decisão clínica. "
        "Todas as sugestões requerem validação do médico assistente.*"
    )

    resposta_final = "\n".join(secoes)

    etapas = state.get("etapas_executadas", [])
    etapas.append("gerar_resposta_final")

    return {
        **state,
        "final_response": resposta_final,
        "etapas_executadas": etapas,
    }


# ============================================================
# CONDIÇÃO DE ROTEAMENTO
# ============================================================

def deve_notificar_equipe(state: PatientState) -> str:
    """
    Função de roteamento condicional do grafo.

    Verifica se há alertas pendentes para decidir o próximo nó.

    Parâmetros:
        state: Estado atual do paciente

    Retorna:
        Nome do próximo nó a executar
    """
    alertas = state.get("alerts", [])

    if alertas:
        logger.info(f"[Grafo] Roteamento → notificar_equipe_medica ({len(alertas)} alertas)")
        return "notificar_equipe_medica"
    else:
        logger.info("[Grafo] Roteamento → gerar_resposta_final (sem alertas)")
        return "gerar_resposta_final"


# ============================================================
# CONSTRUÇÃO DO GRAFO
# ============================================================

def construir_grafo_clinico() -> StateGraph:
    """
    Constrói e compila o grafo de decisão clínica do LangGraph.

    Define todos os nós, arestas e condicionais do fluxo automatizado.

    Retorna:
        Grafo compilado pronto para execução
    """
    logger.info("Construindo grafo LangGraph de decisão clínica...")

    grafo = StateGraph(PatientState)

    # Adiciona os nós ao grafo
    grafo.add_node("verificar_exames_pendentes", verificar_exames_pendentes)
    grafo.add_node("consultar_historico_clinico", consultar_historico_clinico)
    grafo.add_node("sugerir_conduta_tratamento", sugerir_conduta_tratamento)
    grafo.add_node("validar_seguranca", validar_seguranca)
    grafo.add_node("notificar_equipe_medica", notificar_equipe_medica)
    grafo.add_node("gerar_resposta_final", gerar_resposta_final)

    # Define o ponto de entrada do grafo
    grafo.set_entry_point("verificar_exames_pendentes")

    # Define as arestas sequenciais
    grafo.add_edge("verificar_exames_pendentes", "consultar_historico_clinico")
    grafo.add_edge("consultar_historico_clinico", "sugerir_conduta_tratamento")
    grafo.add_edge("sugerir_conduta_tratamento", "validar_seguranca")

    # Aresta condicional: se há alertas → notificar equipe; senão → resposta final
    grafo.add_conditional_edges(
        "validar_seguranca",
        deve_notificar_equipe,
        {
            "notificar_equipe_medica": "notificar_equipe_medica",
            "gerar_resposta_final": "gerar_resposta_final",
        }
    )

    # Após notificação, vai para resposta final
    grafo.add_edge("notificar_equipe_medica", "gerar_resposta_final")

    # Resposta final encerra o grafo
    grafo.add_edge("gerar_resposta_final", END)

    grafo_compilado = grafo.compile()
    logger.info("Grafo LangGraph compilado com sucesso")

    return grafo_compilado


def processar_paciente(
    patient_id: str,
    symptoms: str,
    pending_exams: str = "",
    history: str = "",
    allergies: str = "Nenhuma conhecida",
    current_medications: str = "Nenhum"
) -> PatientState:
    """
    Executa o fluxo completo do grafo para um paciente.

    Parâmetros:
        patient_id: Identificador do paciente (será anonimizado)
        symptoms: Sintomas e queixas atuais
        pending_exams: Exames pendentes e resultados disponíveis
        history: Histórico clínico relevante
        allergies: Alergias medicamentosas conhecidas
        current_medications: Medicamentos em uso atual

    Retorna:
        Estado final do paciente após processamento completo
    """
    logger.info(f"Iniciando processamento do grafo para paciente {patient_id}")

    # Estado inicial do paciente
    estado_inicial: PatientState = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "pending_exams": pending_exams,
        "history": history,
        "allergies": allergies,
        "current_medications": current_medications,
        "suggested_treatment": "",
        "alerts": [],
        "exam_analysis": "",
        "safety_validated": False,
        "final_response": "",
        "etapas_executadas": [],
    }

    # Constrói e executa o grafo
    grafo = construir_grafo_clinico()

    try:
        estado_final = grafo.invoke(estado_inicial)
        logger.info(
            f"Grafo concluído para paciente {patient_id}. "
            f"Etapas: {estado_final.get('etapas_executadas', [])}"
        )
        return estado_final

    except Exception as erro:
        logger.error(f"Erro no processamento do grafo: {erro}")
        estado_inicial["final_response"] = (
            f"Erro ao processar dados do paciente: {erro}. "
            f"Consulte diretamente o médico assistente."
        )
        return estado_inicial
