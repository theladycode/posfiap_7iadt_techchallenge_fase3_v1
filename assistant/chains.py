"""
chains.py
---------
Define as chains do LangChain para o assistente médico.

Chains disponíveis:
1. clinical_qa_chain      — Responde perguntas clínicas usando RAG
2. exam_review_chain      — Analisa exames pendentes do paciente
3. treatment_chain        — Sugere tratamentos baseados nos protocolos
4. alert_chain            — Gera alertas para a equipe médica

IMPORTANTE: Todas as chains incluem disclaimer obrigatório de que as respostas
são sugestões de apoio clínico e NÃO substituem avaliação médica profissional.
"""

import logging
import os
from typing import Optional

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

from assistant.retriever import obter_retriever_global, buscar_com_scores
from assistant.safety import validador_seguranca
from assistant.logger import logger_auditoria

# ============================================================
# CONSTANTES
# ============================================================
CAMINHO_MODELO_PADRAO = "models/medical-llama3-qlora"
TEMPERATURA_LLM = 0.3      # Baixa temperatura para respostas mais precisas
MAX_TOKENS_RESPOSTA = 512
SYSTEM_PROMPT_BASE = (
    "Você é um assistente de apoio clínico especializado em medicina baseada em evidências. "
    "Suas respostas são sugestões fundamentadas em protocolos médicos e NÃO substituem "
    "a avaliação, diagnóstico e prescrição de um profissional de saúde habilitado. "
    "Sempre indique a fonte do protocolo utilizado na resposta. "
    "Use linguagem técnica, porém clara. Seja objetivo e estruturado."
)

# Configura o logger
logger = logging.getLogger(__name__)


# ============================================================
# TEMPLATES DE PROMPT
# ============================================================

TEMPLATE_CLINICAL_QA = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=f"""{SYSTEM_PROMPT_BASE}

Contexto clínico recuperado dos protocolos:
{{context}}

Histórico da conversa:
{{chat_history}}

Pergunta clínica: {{question}}

Resposta baseada em evidências (indique a fonte no final):"""
)

TEMPLATE_EXAM_REVIEW = PromptTemplate(
    input_variables=["exames", "historico", "queixas"],
    template=f"""{SYSTEM_PROMPT_BASE}

Você está analisando os dados laboratoriais e de imagem de um paciente.

Queixas e sintomas reportados:
{{queixas}}

Histórico clínico relevante:
{{historico}}

Exames pendentes e resultados disponíveis:
{{exames}}

Analise os exames de forma sistematizada e indique:
1. Achados normais e alterados
2. Correlação clínico-laboratorial com os sintomas
3. Exames adicionais sugeridos (se necessário)
4. Principais hipóteses diagnósticas (em ordem de probabilidade)

Lembre-se: Esta é uma análise de apoio. A interpretação definitiva cabe ao médico assistente."""
)

TEMPLATE_TREATMENT = PromptTemplate(
    input_variables=["diagnostico_hipotetico", "historico", "alergias", "medicamentos_em_uso"],
    template=f"""{SYSTEM_PROMPT_BASE}

Você está sugerindo condutas terapêuticas com base em protocolos clínicos.

Hipótese diagnóstica principal:
{{diagnostico_hipotetico}}

Histórico médico relevante:
{{historico}}

Alergias conhecidas:
{{alergias}}

Medicamentos em uso:
{{medicamentos_em_uso}}

Sugira uma conduta terapêutica estruturada incluindo:
1. Tratamento de primeira linha recomendado pelo protocolo
2. Alternativas em caso de contraindicações
3. Pontos de atenção e monitoramento
4. Critérios de encaminhamento ou internação

IMPORTANTE: Todas as sugestões requerem validação e prescrição pelo médico responsável."""
)

TEMPLATE_ALERT = PromptTemplate(
    input_variables=["dados_paciente", "achados_criticos"],
    template=f"""{SYSTEM_PROMPT_BASE}

ANÁLISE DE ALERTA CLÍNICO

Dados do paciente:
{{dados_paciente}}

Achados críticos identificados:
{{achados_criticos}}

Gere um alerta clínico estruturado para a equipe médica incluindo:
1. Nível de urgência (CRÍTICO / ALTO / MODERADO)
2. Achados que justificam o alerta
3. Ações imediatas recomendadas
4. Equipe/especialidade a notificar

Este alerta é gerado automaticamente e requer confirmação da equipe médica."""
)


def carregar_llm(caminho_modelo: str = CAMINHO_MODELO_PADRAO) -> HuggingFacePipeline:
    """
    Carrega o modelo LLM fine-tunado e cria um pipeline HuggingFace.

    Tenta carregar o modelo fine-tunado; se não encontrar, usa um modelo
    menor de demonstração para não bloquear a execução.

    Parâmetros:
        caminho_modelo: Caminho para o modelo fine-tunado

    Retorna:
        Objeto HuggingFacePipeline compatível com LangChain
    """
    import os
    from pathlib import Path

    token_hf = os.environ.get("HUGGINGFACE_TOKEN")
    modelo_env = os.environ.get("MODEL_PATH", caminho_modelo)

    # Verifica se o modelo fine-tunado existe
    if not Path(modelo_env).exists():
        logger.warning(
            f"Modelo fine-tunado não encontrado em '{modelo_env}'. "
            f"Usando modelo de demonstração (microsoft/phi-2)."
        )
        nome_modelo = "microsoft/phi-2"
    else:
        nome_modelo = modelo_env
        logger.info(f"Carregando modelo fine-tunado: {nome_modelo}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Detecta se é um adaptador LoRA local
    adapter_config_path = Path(nome_modelo) / "adapter_config.json"
    if adapter_config_path.exists():
        import json as _json
        with open(adapter_config_path) as f:
            adapter_cfg = _json.load(f)
        nome_base = adapter_cfg.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.1")
        logger.info(f"Adaptador LoRA detectado. Modelo base: {nome_base}")
    else:
        nome_base = nome_modelo

    tokenizador = AutoTokenizer.from_pretrained(
        nome_modelo,
        token=token_hf,
        trust_remote_code=True
    )
    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token

    modelo_base = AutoModelForCausalLM.from_pretrained(
        nome_base,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        token=token_hf,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_config_path.exists():
        modelo = PeftModel.from_pretrained(modelo_base, nome_modelo)
        logger.info("Adaptadores LoRA aplicados ao modelo base")
    else:
        modelo = modelo_base

    pipe = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizador,
        max_new_tokens=MAX_TOKENS_RESPOSTA,
        temperature=TEMPERATURA_LLM,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizador.eos_token_id,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("LLM carregado e pronto para uso nas chains")
    return llm


class AssistenteMedico:
    """
    Orquestra todas as chains do assistente médico.

    Gerencia o ciclo de vida do LLM, memória conversacional e
    roteamento de perguntas para as chains apropriadas.

    Atributos:
        llm: Modelo de linguagem carregado
        retriever: Sistema RAG para busca semântica
        memoria: Buffer de memória conversacional
        chains: Dicionário com as chains disponíveis
    """

    def __init__(self, caminho_modelo: str = CAMINHO_MODELO_PADRAO) -> None:
        """
        Inicializa o assistente médico com todas as chains configuradas.

        Parâmetros:
            caminho_modelo: Caminho para o modelo LLM fine-tunado
        """
        logger.info("Inicializando Assistente Médico...")

        self.llm = carregar_llm(caminho_modelo)
        self.retriever = obter_retriever_global()
        self.memoria = self._criar_memoria()
        self.chains = self._inicializar_chains()

        logger.info("Assistente Médico pronto para uso")

    def _criar_memoria(self) -> list:
        """
        Cria a lista de histórico conversacional.

        Retorna:
            Lista vazia para acumular mensagens da conversa
        """
        return []

    def _inicializar_chains(self) -> dict:
        """
        Inicializa e retorna todas as chains disponíveis no assistente (LCEL).

        Retorna:
            Dicionário mapeando nomes para objetos de chain
        """
        parser = StrOutputParser()
        chains = {}

        # Chain 1: Consulta Clínica com RAG (LCEL)
        def _formatar_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        chains["clinical_qa"] = (
            {
                "context": self.retriever | _formatar_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: "\n".join(
                    f"{m['role']}: {m['content']}" for m in self.memoria
                ),
            }
            | TEMPLATE_CLINICAL_QA
            | self.llm
            | parser
        )

        # Chain 2: Análise de Exames (LCEL)
        chains["exam_review"] = TEMPLATE_EXAM_REVIEW | self.llm | parser

        # Chain 3: Sugestão de Tratamento (LCEL)
        chains["treatment"] = TEMPLATE_TREATMENT | self.llm | parser

        # Chain 4: Geração de Alertas (LCEL)
        chains["alert"] = TEMPLATE_ALERT | self.llm | parser

        logger.info(f"Chains inicializadas: {list(chains.keys())}")
        return chains

    def responder_pergunta_clinica(self, pergunta: str) -> dict:
        """
        Responde perguntas clínicas gerais usando RAG + histórico conversacional.

        Parâmetros:
            pergunta: Pergunta clínica do usuário

        Retorna:
            Dicionário com resposta, fontes e metadados de segurança
        """
        logger.info(f"Processando pergunta clínica (comprimento: {len(pergunta)} chars)")

        # Valida a pergunta antes de processar
        if not validador_seguranca.verificar_pergunta(pergunta):
            resposta_bloqueada = (
                "Esta pergunta não pôde ser processada pelo sistema de apoio clínico. "
                "Por favor, reformule sua pergunta ou consulte diretamente um profissional de saúde."
            )
            logger_auditoria.registrar_interacao(
                pergunta_usuario=pergunta,
                resposta_assistente=resposta_bloqueada,
                chain_utilizada="clinical_qa",
                flag_alerta=True
            )
            return {"resposta": resposta_bloqueada, "fontes": [], "flag_alerta": True}

        try:
            resposta_bruta = self.chains["clinical_qa"].invoke(pergunta)
            if not isinstance(resposta_bruta, str):
                resposta_bruta = str(resposta_bruta)

            # Recupera documentos com scores para auditoria enriquecida
            docs_com_scores = buscar_com_scores(pergunta)
            fontes = list({d["fonte"] for d in docs_com_scores})

            # Motivo: protocolo mais relevante (maior score)
            motivo = None
            if docs_com_scores:
                melhor = max(docs_com_scores, key=lambda d: d["score_similaridade"])
                motivo = (
                    f"Protocolo mais relevante: '{melhor['fonte']}' "
                    f"(score={melhor['score_similaridade']:.2f}). "
                    f"{len(docs_com_scores)} documentos recuperados do índice FAISS."
                )

            # Atualiza histórico conversacional
            self.memoria.append({"role": "user", "content": pergunta})
            self.memoria.append({"role": "assistant", "content": resposta_bruta})

            # Valida e sanitiza a resposta
            validacao = validador_seguranca.validar_resposta(
                pergunta=pergunta,
                resposta=resposta_bruta,
                chain_utilizada="clinical_qa"
            )

            # Registra na auditoria com dados de RAG
            scores = [d["score_similaridade"] for d in docs_com_scores]
            id_interacao = logger_auditoria.registrar_interacao(
                pergunta_usuario=pergunta,
                resposta_assistente=validacao.resposta_final,
                chain_utilizada="clinical_qa",
                fontes_citadas=fontes,
                flag_alerta=validacao.flag_alerta,
                documentos_rag=docs_com_scores,
                motivo_resposta=motivo,
                metadata_adicional={
                    "num_docs_recuperados": len(docs_com_scores),
                    "score_maximo": round(max(scores), 4) if scores else None,
                    "score_minimo": round(min(scores), 4) if scores else None,
                    "score_medio": round(sum(scores) / len(scores), 4) if scores else None,
                    "tamanho_historico": len(self.memoria),
                    "seguranca_modificou_resposta": validacao.resposta_final != resposta_bruta,
                    "e_emergencia": validacao.e_emergencia,
                },
            )

            return {
                "id_interacao": id_interacao,
                "resposta": validacao.resposta_final,
                "fontes": fontes,
                "flag_alerta": validacao.flag_alerta,
                "e_emergencia": validacao.e_emergencia,
                "documentos_rag": docs_com_scores,
                "motivo_resposta": motivo,
            }

        except Exception as erro:
            logger.error(f"Erro na chain clinical_qa: {erro}")
            return {
                "resposta": "Ocorreu um erro ao processar sua pergunta. Tente novamente.",
                "fontes": [],
                "flag_alerta": False,
                "erro": str(erro)
            }

    def analisar_exames(
        self,
        exames: str,
        historico: str = "Não informado",
        queixas: str = "Não informado"
    ) -> dict:
        """
        Analisa exames laboratoriais e de imagem de um paciente.

        Parâmetros:
            exames: Descrição dos exames e resultados
            historico: Histórico clínico do paciente
            queixas: Sintomas e queixas atuais

        Retorna:
            Dicionário com análise e metadados
        """
        logger.info("Processando análise de exames via exam_review chain")

        try:
            resposta_bruta = self.chains["exam_review"].invoke({
                "exames": exames,
                "historico": historico,
                "queixas": queixas,
            })
            if not isinstance(resposta_bruta, str):
                resposta_bruta = str(resposta_bruta)
            validacao = validador_seguranca.validar_resposta(
                pergunta=f"Análise de exames: {exames[:100]}",
                resposta=resposta_bruta,
                chain_utilizada="exam_review"
            )

            id_interacao = logger_auditoria.registrar_interacao(
                pergunta_usuario=f"Análise de exames: {exames}",
                resposta_assistente=validacao.resposta_final,
                chain_utilizada="exam_review",
                flag_alerta=validacao.flag_alerta,
                metadata_adicional={
                    "historico_fornecido": historico != "Não informado",
                    "queixas_fornecidas": queixas != "Não informado",
                    "tamanho_exames": len(exames),
                    "seguranca_modificou_resposta": validacao.resposta_final != resposta_bruta,
                },
            )

            return {"id_interacao": id_interacao, "resposta": validacao.resposta_final, "flag_alerta": validacao.flag_alerta}

        except Exception as erro:
            logger.error(f"Erro na chain exam_review: {erro}")
            return {"resposta": f"Erro na análise de exames: {erro}", "flag_alerta": False}

    def sugerir_tratamento(
        self,
        diagnostico_hipotetico: str,
        historico: str = "Não informado",
        alergias: str = "Nenhuma conhecida",
        medicamentos_em_uso: str = "Nenhum"
    ) -> dict:
        """
        Sugere condutas terapêuticas baseadas em protocolos clínicos.

        Parâmetros:
            diagnostico_hipotetico: Hipótese diagnóstica principal
            historico: Histórico médico do paciente
            alergias: Alergias medicamentosas conhecidas
            medicamentos_em_uso: Medicamentos em uso atual

        Retorna:
            Dicionário com sugestão de tratamento e metadados
        """
        logger.info(f"Gerando sugestão de tratamento para: {diagnostico_hipotetico}")

        try:
            resposta_bruta = self.chains["treatment"].invoke({
                "diagnostico_hipotetico": diagnostico_hipotetico,
                "historico": historico,
                "alergias": alergias,
                "medicamentos_em_uso": medicamentos_em_uso,
            })
            if not isinstance(resposta_bruta, str):
                resposta_bruta = str(resposta_bruta)
            validacao = validador_seguranca.validar_resposta(
                pergunta=f"Tratamento para: {diagnostico_hipotetico}",
                resposta=resposta_bruta,
                chain_utilizada="treatment"
            )

            id_interacao = logger_auditoria.registrar_interacao(
                pergunta_usuario=f"Sugestão de tratamento: {diagnostico_hipotetico}",
                resposta_assistente=validacao.resposta_final,
                chain_utilizada="treatment",
                flag_alerta=validacao.flag_alerta,
                metadata_adicional={
                    "diagnostico": diagnostico_hipotetico[:100],
                    "tem_alergias": alergias not in ("Nenhuma conhecida", ""),
                    "tem_medicamentos_em_uso": medicamentos_em_uso not in ("Nenhum", ""),
                    "historico_fornecido": historico != "Não informado",
                    "seguranca_modificou_resposta": validacao.resposta_final != resposta_bruta,
                },
            )

            return {"id_interacao": id_interacao, "resposta": validacao.resposta_final, "flag_alerta": validacao.flag_alerta}

        except Exception as erro:
            logger.error(f"Erro na chain treatment: {erro}")
            return {"resposta": f"Erro na geração de tratamento: {erro}", "flag_alerta": False}

    def gerar_alerta(self, dados_paciente: str, achados_criticos: str) -> dict:
        """
        Gera alerta clínico estruturado para a equipe médica.

        Parâmetros:
            dados_paciente: Informações do paciente
            achados_criticos: Achados laboratoriais ou clínicos críticos

        Retorna:
            Dicionário com alerta e metadados
        """
        logger.info("Gerando alerta clínico via alert chain")

        try:
            resposta_bruta = self.chains["alert"].invoke({
                "dados_paciente": dados_paciente,
                "achados_criticos": achados_criticos,
            })
            if not isinstance(resposta_bruta, str):
                resposta_bruta = str(resposta_bruta)
            validacao = validador_seguranca.validar_resposta(
                pergunta=f"Alerta: {achados_criticos[:100]}",
                resposta=resposta_bruta,
                chain_utilizada="alert"
            )

            id_interacao = logger_auditoria.registrar_interacao(
                pergunta_usuario=f"Alerta clínico: {achados_criticos}",
                resposta_assistente=validacao.resposta_final,
                chain_utilizada="alert",
                flag_alerta=True,
                metadata_adicional={
                    "tamanho_achados_criticos": len(achados_criticos),
                    "tamanho_dados_paciente": len(dados_paciente),
                    "seguranca_modificou_resposta": validacao.resposta_final != resposta_bruta,
                },
            )

            return {"id_interacao": id_interacao, "resposta": validacao.resposta_final, "flag_alerta": True}

        except Exception as erro:
            logger.error(f"Erro na chain alert: {erro}")
            return {"resposta": f"Erro na geração de alerta: {erro}", "flag_alerta": True}

    def limpar_memoria(self) -> None:
        """
        Limpa o histórico da conversa atual, iniciando nova sessão.
        """
        self.memoria.clear()
        # Reinicializa chains para atualizar a referência ao histórico
        self.chains = self._inicializar_chains()
        logger.info("Memória conversacional limpa — nova sessão iniciada")


# Instância global do assistente (lazy loading)
_assistente_global: Optional[AssistenteMedico] = None


def obter_assistente() -> AssistenteMedico:
    """
    Retorna a instância global do assistente médico, criando se necessário.

    Retorna:
        Instância compartilhada do AssistenteMedico
    """
    global _assistente_global

    if _assistente_global is None:
        caminho_modelo = os.environ.get("MODEL_PATH", CAMINHO_MODELO_PADRAO)
        _assistente_global = AssistenteMedico(caminho_modelo)

    return _assistente_global
