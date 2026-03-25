"""
api.py
------
API REST do Assistente Médico Virtual com documentação Swagger automática.

Endpoints disponíveis:
- POST /chat          — Consulta clínica conversacional (RAG + LLM)
- POST /exams         — Análise de exames laboratoriais
- POST /treatment     — Sugestão de conduta terapêutica
- POST /alert         — Geração de alerta clínico
- POST /analyze       — Análise completa via LangGraph
- GET  /health        — Status da API
- GET  /stats         — Estatísticas de uso

Uso:
    uvicorn interface.api:app --reload --port 8000
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Garante que a raiz do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from assistant.chains import obter_assistente
from assistant.logger import logger_auditoria

# ============================================================
# CONFIGURAÇÃO DO LOGGER
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================
# INSTÂNCIA FASTAPI
# ============================================================
app = FastAPI(
    title="Assistente Médico Virtual — API",
    description=(
        "API REST do sistema de apoio à decisão clínica baseado em LLM fine-tunado (QLoRA) "
        "com RAG (FAISS + SentenceTransformers) e orquestração via LangChain/LangGraph.\n\n"
        "**AVISO:** Este sistema é um assistente de apoio clínico. As respostas **NÃO substituem** "
        "a avaliação de um profissional de saúde habilitado. Em emergências: **SAMU 192**."
    ),
    version="1.0.0",
    contact={
        "name": "FIAP PosTech — Projeto de Sucesso",
        "url": "https://huggingface.co/theladycode/NEURIX",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SCHEMAS DE REQUEST / RESPONSE
# ============================================================

class ChatRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "pergunta": "Qual o tratamento de primeira linha para hipertensão arterial estágio 1?"
    }}}

    pergunta: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        title="[OBRIGATÓRIO] Pergunta clínica",
        description="Pergunta clínica a ser respondida (3–2000 caracteres)",
    )

class ChatResponse(BaseModel):
    id_interacao: str = Field(..., description="ID único da interação — use em GET /audit/{id_interacao}")
    resposta: str = Field(..., description="Resposta gerada pelo assistente")
    fontes: list[str] = Field(default=[], description="Fontes e protocolos citados")
    flag_alerta: bool = Field(..., description="True se o módulo de segurança interveio")
    e_emergencia: bool = Field(default=False, description="True se situação de emergência detectada")


class ExameRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "exames": "Hemograma: Hb 9g/dL, Leucócitos 15.000/mm³, PCR 120mg/L, Plaquetas 420.000",
        "historico": "Paciente com DM2 e HAS em uso de metformina e losartana",
        "queixas": "Febre há 3 dias, calafrios, disúria"
    }}}

    exames: str = Field(
        ...,
        min_length=5,
        title="[OBRIGATÓRIO] Resultados dos exames",
        description="Descrição dos exames e resultados disponíveis",
    )
    historico: str = Field(
        default="Não informado",
        title="[opcional] Histórico clínico",
        description="Histórico clínico relevante do paciente",
    )
    queixas: str = Field(
        default="Não informado",
        title="[opcional] Queixas e sintomas",
        description="Sintomas e queixas atuais do paciente",
    )

class ExameResponse(BaseModel):
    id_interacao: str = Field(..., description="ID único da interação — use em GET /audit/{id_interacao}")
    analise: str = Field(..., description="Análise clínica dos exames")
    flag_alerta: bool = Field(..., description="True se achados críticos detectados")


class TratamentoRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "diagnostico_hipotetico": "Pneumonia adquirida na comunidade (PAC) moderada",
        "historico": "Paciente tabagista, DPOC leve",
        "alergias": "Alergia a penicilina",
        "medicamentos_em_uso": "Salbutamol spray SOS"
    }}}

    diagnostico_hipotetico: str = Field(
        ...,
        min_length=3,
        title="[OBRIGATÓRIO] Hipótese diagnóstica",
        description="Hipótese diagnóstica principal para guiar a conduta terapêutica",
    )
    historico: str = Field(
        default="Não informado",
        title="[opcional] Histórico médico",
        description="Histórico médico relevante do paciente",
    )
    alergias: str = Field(
        default="Nenhuma conhecida",
        title="[opcional] Alergias medicamentosas",
        description="Alergias a medicamentos conhecidas",
    )
    medicamentos_em_uso: str = Field(
        default="Nenhum",
        title="[opcional] Medicamentos em uso",
        description="Medicamentos em uso atual pelo paciente",
    )

class TratamentoResponse(BaseModel):
    id_interacao: str = Field(..., description="ID único da interação — use em GET /audit/{id_interacao}")
    sugestao: str = Field(..., description="Sugestão de conduta terapêutica baseada em protocolos")
    flag_alerta: bool = Field(..., description="True se atenção especial requerida")


class AlertaRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "dados_paciente": "Paciente 65 anos, hipertenso, diabético, admitido com dor torácica há 2h",
        "achados_criticos": "Troponina 15x acima do limite, ST elevado em V1-V4, PA 90/60 mmHg"
    }}}

    dados_paciente: str = Field(
        ...,
        min_length=5,
        title="[OBRIGATÓRIO] Dados do paciente",
        description="Informações clínicas e demográficas do paciente",
    )
    achados_criticos: str = Field(
        ...,
        min_length=5,
        title="[OBRIGATÓRIO] Achados críticos",
        description="Achados laboratoriais ou clínicos críticos que justificam o alerta",
    )

class AlertaResponse(BaseModel):
    id_interacao: str = Field(..., description="ID único da interação — use em GET /audit/{id_interacao}")
    alerta: str = Field(..., description="Alerta clínico estruturado para a equipe")
    flag_alerta: bool = Field(default=True, description="Sempre True para alertas")


class AnalisePacienteRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "patient_id": "PAC-2024-001",
        "sintomas": "Febre 39°C há 3 dias, tosse produtiva com expectoração amarelada, dispneia aos médios esforços",
        "exames": "RX tórax: consolidação em lobo inferior direito. PCR 85mg/L, Leucócitos 14.500",
        "historico": "Tabagista 20 anos/maço, sem comorbidades conhecidas",
        "alergias": "Nenhuma conhecida",
        "medicamentos": "Nenhum"
    }}}

    patient_id: str = Field(
        default="PAC-001",
        title="[opcional] ID do paciente",
        description="Identificador do paciente — será anonimizado no log de auditoria",
    )
    sintomas: str = Field(
        ...,
        min_length=5,
        title="[OBRIGATÓRIO] Sintomas e queixas",
        description="Descrição detalhada dos sintomas e queixas atuais do paciente",
    )
    exames: str = Field(
        default="Não disponíveis",
        title="[opcional] Resultados de exames",
        description="Resultados de exames laboratoriais ou de imagem disponíveis",
    )
    historico: str = Field(
        default="Não informado",
        title="[opcional] Histórico clínico",
        description="Comorbidades, cirurgias prévias e histórico médico relevante",
    )
    alergias: str = Field(
        default="Nenhuma conhecida",
        title="[opcional] Alergias",
        description="Alergias medicamentosas conhecidas",
    )
    medicamentos: str = Field(
        default="Nenhum",
        title="[opcional] Medicamentos em uso",
        description="Lista de medicamentos em uso atual",
    )

class AnalisePacienteResponse(BaseModel):
    analise_exames: str = Field(..., description="Análise dos exames disponíveis")
    sugestao_tratamento: str = Field(..., description="Sugestão terapêutica")
    alerta_gerado: str = Field(..., description="Alerta clínico se aplicável")
    tem_alerta: bool = Field(..., description="True se alertas críticos foram gerados")


class HealthResponse(BaseModel):
    status: str
    versao: str
    modelo: str
    sessao_logger: str


class StatsResponse(BaseModel):
    total_interacoes: int
    total_alertas: int
    percentual_alertas: float
    chains_mais_usadas: list


class DocumentoRAG(BaseModel):
    posicao_ranking: int = Field(..., description="Posição no ranking de relevância (1 = mais relevante)")
    conteudo: str = Field(..., description="Trecho do documento recuperado (até 300 chars)")
    fonte: str = Field(..., description="Protocolo ou fonte do documento")
    score_similaridade: float = Field(..., description="Score de similaridade semântica (0-1, maior = mais relevante)")


class InteracaoAuditoria(BaseModel):
    id_interacao: str
    id_sessao: str
    timestamp: str
    chain_utilizada: str
    pergunta_anonimizada: str
    resposta_gerada: str
    comprimento_pergunta: int
    comprimento_resposta: int
    fontes_citadas: list[str]
    flag_alerta_seguranca: bool
    documentos_rag: list[DocumentoRAG] = Field(
        default=[],
        description="Documentos recuperados do índice FAISS com scores de similaridade"
    )
    confianca_estimada: Optional[float] = Field(
        default=None,
        description="Confiança estimada (0-1): média dos scores RAG, penalizada se flag de alerta ativo"
    )
    motivo_resposta: Optional[str] = Field(
        default=None,
        description="Explicação do motivo da resposta: protocolo mais relevante e quantidade de docs usados"
    )
    metadata: dict


class AuditoriaResponse(BaseModel):
    total: int
    interacoes: list[InteracaoAuditoria]


# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificação de saúde da API",
    tags=["Sistema"],
)
def health_check():
    """
    Retorna o status atual da API, versão e informações da sessão de logging.
    Útil para monitoramento e verificação de disponibilidade.
    """
    return HealthResponse(
        status="ok",
        versao="1.0.0",
        modelo=os.environ.get("MODEL_PATH", "models/medical-llama3-qlora"),
        sessao_logger=logger_auditoria.id_sessao,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Estatísticas de uso do sistema",
    tags=["Sistema"],
)
def obter_estatisticas():
    """
    Retorna métricas de uso: total de interações, alertas gerados e
    chains mais utilizadas. Baseado nos logs de auditoria persistidos.
    """
    stats = logger_auditoria.obter_estatisticas()
    return StatsResponse(
        total_interacoes=stats.get("total_interacoes", 0),
        total_alertas=stats.get("total_alertas", 0),
        percentual_alertas=stats.get("percentual_alertas", 0.0),
        chains_mais_usadas=stats.get("chains_mais_usadas", []),
    )


@app.get(
    "/audit",
    response_model=AuditoriaResponse,
    summary="Log de auditoria das interações",
    tags=["Sistema"],
)
def obter_auditoria(
    limite: int = 20,
    apenas_alertas: bool = False,
    sessao: Optional[str] = None,
):
    """
    Retorna o log de auditoria das interações registradas.

    Parâmetros de filtro:
    - **limite**: número máximo de registros a retornar (padrão: 20, máx: 500)
    - **apenas_alertas**: se True, retorna apenas interações com flag de segurança ativa
    - **sessao**: filtra por ID de sessão específico
    """
    limite = min(limite, 500)
    interacoes = logger_auditoria.recuperar_interacoes_recentes(num_interacoes=limite)

    if apenas_alertas:
        interacoes = [i for i in interacoes if i.get("flag_alerta_seguranca")]

    if sessao:
        interacoes = [i for i in interacoes if i.get("id_sessao") == sessao]

    return AuditoriaResponse(
        total=len(interacoes),
        interacoes=[InteracaoAuditoria(**i) for i in interacoes],
    )


@app.get(
    "/audit/{id_interacao}",
    response_model=InteracaoAuditoria,
    summary="Busca interação por ID",
    tags=["Sistema"],
)
def obter_interacao_por_id(id_interacao: str):
    """
    Retorna os dados completos de uma interação específica pelo seu `id_interacao`.

    Inclui documentos RAG recuperados, score de confiança e motivo da resposta.
    Retorna **404** se o ID não for encontrado.
    """
    todas = logger_auditoria.recuperar_interacoes_recentes(num_interacoes=10000)
    for interacao in todas:
        if interacao.get("id_interacao") == id_interacao:
            return InteracaoAuditoria(**interacao)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Interação '{id_interacao}' não encontrada.",
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Consulta clínica conversacional",
    tags=["Assistente"],
)
def chat_clinico(req: ChatRequest):
    """
    Responde perguntas clínicas usando RAG (recuperação semântica de protocolos)
    combinado com o LLM fine-tunado.

    O histórico conversacional é mantido na sessão para perguntas de acompanhamento.
    O módulo de segurança valida a pergunta e sanitiza a resposta antes de retornar.
    """
    try:
        assistente = obter_assistente()
        resultado = assistente.responder_pergunta_clinica(req.pergunta)
        return ChatResponse(
            id_interacao=resultado.get("id_interacao", ""),
            resposta=resultado.get("resposta", ""),
            fontes=resultado.get("fontes", []),
            flag_alerta=resultado.get("flag_alerta", False),
            e_emergencia=resultado.get("e_emergencia", False),
        )
    except Exception as e:
        logger.error(f"Erro em /chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/exams",
    response_model=ExameResponse,
    summary="Análise de exames laboratoriais e de imagem",
    tags=["Assistente"],
)
def analisar_exames(req: ExameRequest):
    """
    Analisa resultados de exames laboratoriais e de imagem, correlacionando
    com os sintomas e histórico clínico informados.

    Retorna achados sistematizados, hipóteses diagnósticas e sugestões de
    exames complementares quando aplicável.
    """
    try:
        assistente = obter_assistente()
        resultado = assistente.analisar_exames(
            exames=req.exames,
            historico=req.historico,
            queixas=req.queixas,
        )
        return ExameResponse(
            id_interacao=resultado.get("id_interacao", ""),
            analise=resultado.get("resposta", ""),
            flag_alerta=resultado.get("flag_alerta", False),
        )
    except Exception as e:
        logger.error(f"Erro em /exams: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/treatment",
    response_model=TratamentoResponse,
    summary="Sugestão de conduta terapêutica",
    tags=["Assistente"],
)
def sugerir_tratamento(req: TratamentoRequest):
    """
    Gera sugestão de conduta terapêutica baseada em protocolos clínicos,
    considerando hipótese diagnóstica, histórico, alergias e medicações em uso.

    **Todas as sugestões requerem validação e prescrição pelo médico responsável.**
    """
    try:
        assistente = obter_assistente()
        resultado = assistente.sugerir_tratamento(
            diagnostico_hipotetico=req.diagnostico_hipotetico,
            historico=req.historico,
            alergias=req.alergias,
            medicamentos_em_uso=req.medicamentos_em_uso,
        )
        return TratamentoResponse(
            id_interacao=resultado.get("id_interacao", ""),
            sugestao=resultado.get("resposta", ""),
            flag_alerta=resultado.get("flag_alerta", False),
        )
    except Exception as e:
        logger.error(f"Erro em /treatment: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/alert",
    response_model=AlertaResponse,
    summary="Geração de alerta clínico",
    tags=["Assistente"],
)
def gerar_alerta(req: AlertaRequest):
    """
    Gera um alerta clínico estruturado para a equipe médica com base nos
    dados do paciente e achados críticos identificados.

    O alerta inclui: nível de urgência, achados que o justificam,
    ações imediatas recomendadas e especialidade a notificar.
    """
    try:
        assistente = obter_assistente()
        resultado = assistente.gerar_alerta(
            dados_paciente=req.dados_paciente,
            achados_criticos=req.achados_criticos,
        )
        return AlertaResponse(
            id_interacao=resultado.get("id_interacao", ""),
            alerta=resultado.get("resposta", ""),
            flag_alerta=True,
        )
    except Exception as e:
        logger.error(f"Erro em /alert: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/analyze",
    response_model=AnalisePacienteResponse,
    summary="Análise completa do paciente via LangGraph",
    tags=["Assistente"],
)
def analisar_paciente(req: AnalisePacienteRequest):
    """
    Executa o fluxo completo de análise clínica via LangGraph:
    1. Triagem inicial dos sintomas
    2. Análise dos exames disponíveis
    3. Sugestão de conduta terapêutica
    4. Geração de alerta se achados críticos

    Retorna relatório consolidado com todas as etapas.
    """
    try:
        assistente = obter_assistente()

        analise = assistente.analisar_exames(
            exames=req.exames,
            historico=req.historico,
            queixas=req.sintomas,
        )
        tratamento = assistente.sugerir_tratamento(
            diagnostico_hipotetico=f"Baseado em: {req.sintomas[:200]}",
            historico=req.historico,
            alergias=req.alergias,
            medicamentos_em_uso=req.medicamentos,
        )
        alerta = assistente.gerar_alerta(
            dados_paciente=f"ID: {req.patient_id} | {req.historico}",
            achados_criticos=req.sintomas,
        )

        return AnalisePacienteResponse(
            analise_exames=analise.get("resposta", ""),
            sugestao_tratamento=tratamento.get("resposta", ""),
            alerta_gerado=alerta.get("resposta", ""),
            tem_alerta=analise.get("flag_alerta", False) or tratamento.get("flag_alerta", False),
        )
    except Exception as e:
        logger.error(f"Erro em /analyze: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ============================================================
# PONTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    import uvicorn
    porta = int(os.environ.get("API_PORT", 8000))
    logger.info(f"Iniciando API na porta {porta}...")
    logger.info(f"Swagger UI: http://localhost:{porta}/docs")
    uvicorn.run("interface.api:app", host="0.0.0.0", port=porta, reload=False)
