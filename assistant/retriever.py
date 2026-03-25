"""
retriever.py
------------
Implementa o sistema de RAG (Retrieval-Augmented Generation) para o assistente médico.

O retriever indexa documentos médicos (protocolos, diretrizes, artigos) em um
índice vetorial FAISS e realiza buscas semânticas para enriquecer as respostas
do LLM com informações contextuais relevantes.

Fluxo RAG:
1. Documentos médicos são carregados e divididos em chunks
2. Cada chunk é vetorizado com SentenceTransformers
3. Os vetores são indexados no FAISS
4. Na consulta, o texto do usuário é vetorizado e os chunks mais próximos são recuperados
5. Os chunks recuperados são passados como contexto ao LLM
"""

import logging
import os
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================================
# CONSTANTES
# ============================================================
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CAMINHO_INDICE_PADRAO = "data/processed/faiss_index"
CAMINHO_DOCS_PADRAO = "data/raw"
TAMANHO_CHUNK = 800
OVERLAP_CHUNK = 100
NUM_DOCUMENTOS_RECUPERADOS = 4

# Configura o logger
logger = logging.getLogger(__name__)


def criar_documentos_medicos_exemplo() -> list[Document]:
    """
    Cria documentos médicos de exemplo para popular o índice RAG inicial.

    Na produção, esses documentos seriam substituídos por protocolos
    clínicos reais, artigos científicos e diretrizes médicas.

    Retorna:
        Lista de objetos Document do LangChain
    """
    conteudos_medicos = [
        {
            "conteudo": """
Protocolo de Manejo da Hipertensão Arterial Sistêmica
Fonte: Diretriz Brasileira de Hipertensão Arterial 2020

Diagnóstico:
- Hipertensão estágio 1: PAS 140-159 mmHg ou PAD 90-99 mmHg
- Hipertensão estágio 2: PAS 160-179 mmHg ou PAD 100-109 mmHg
- Hipertensão estágio 3: PAS ≥ 180 mmHg ou PAD ≥ 110 mmHg

Tratamento não farmacológico (indicado para todos os estágios):
- Redução do consumo de sal (< 5g/dia)
- Dieta DASH (rica em frutas, vegetais, laticínios desnatados)
- Atividade física aeróbica: 150 min/semana intensidade moderada
- Redução do peso corporal (meta: IMC < 25 kg/m²)
- Cessação do tabagismo
- Moderação no consumo de álcool (< 14 doses/semana homens, < 8 mulheres)

Classes farmacológicas de primeira linha:
- Inibidores da ECA (ex: enalapril, ramipril)
- Bloqueadores do receptor de angiotensina (ex: losartana, valsartana)
- Bloqueadores dos canais de cálcio (ex: anlodipino, nifedipino)
- Diuréticos tiazídicos (ex: hidroclorotiazida, clortalidona)
- Beta-bloqueadores (ex: atenolol, carvedilol) — quando há indicação específica
            """,
            "fonte": "Diretriz Brasileira de Hipertensão 2020",
            "especialidade": "Cardiologia"
        },
        {
            "conteudo": """
Protocolo de Diabetes Mellitus Tipo 2
Fonte: Diretrizes da Sociedade Brasileira de Diabetes 2023

Critérios diagnósticos:
- Glicemia de jejum ≥ 126 mg/dL (em 2 ocasiões)
- Glicemia 2h pós-TOTG ≥ 200 mg/dL
- HbA1c ≥ 6,5%
- Glicemia aleatória ≥ 200 mg/dL com sintomas

Metas glicêmicas:
- HbA1c < 7% (em geral)
- Glicemia de jejum: 80-130 mg/dL
- Glicemia pós-prandial: < 180 mg/dL

Tratamento:
- Primeira linha: metformina 500-2000 mg/dia (salvo contraindicação)
- Segunda linha: considerar DCV estabelecida → agonista GLP-1 ou iSGLT2
- Insulinoterapia: quando HbA1c > 10% ou falha da terapia oral
- Monitoramento: HbA1c a cada 3 meses (descompensado) ou 6 meses (controlado)

Prevenção de complicações:
- Controle de PA (meta < 130/80 mmHg)
- Estatina se risco cardiovascular elevado
- IECA/BRA se proteinúria
- Avaliação anual: fundo de olho, microalbuminúria, neuropatia, pé diabético
            """,
            "fonte": "Diretrizes SBD 2023",
            "especialidade": "Endocrinologia"
        },
        {
            "conteudo": """
Protocolo de Sepse — Surviving Sepsis Campaign 2021

Definição (Sepse-3):
- Sepse: disfunção orgânica ameaçadora à vida por resposta desregulada à infecção
- Critério operacional: aumento ≥ 2 pontos no escore SOFA
- Choque séptico: sepse + vasopressor para PAM ≥ 65 mmHg + lactato > 2 mmol/L

Pacote de 1 hora (Hour-1 Bundle):
1. Dosar lactato sérico (redosar se > 2 mmol/L)
2. Hemoculturas antes de antibióticos
3. Antibioticoterapia de amplo espectro
4. Cristaloide 30 mL/kg se hipotensão ou lactato ≥ 4 mmol/L
5. Vasopressor (noradrenalina) se hipotensão durante/após ressuscitação

Antibioticoterapia:
- Iniciar em até 1 hora do reconhecimento
- Cobertura empírica ampla baseada no foco suspeito
- Desescalonamento em 48-72h conforme culturas

Controle do foco infeccioso:
- Identificar e controlar foco em até 6-12 horas
- Drenagem de abscessos, remoção de dispositivos infectados

Suporte orgânico:
- VM protetora se SDRA (VC 6 mL/kg peso predito)
- Terapia renal substitutiva se IRA grave
- Controle glicêmico: 140-180 mg/dL
- Profilaxia de TEV e úlcera de estresse
            """,
            "fonte": "Surviving Sepsis Campaign 2021",
            "especialidade": "Medicina Intensiva"
        },
        {
            "conteudo": """
Protocolo de Dor Torácica Aguda no Pronto-Socorro

Avaliação inicial (primeiros 10 minutos):
1. ECG de 12 derivações imediato
2. Acesso venoso + analgesia
3. Monitoração cardíaca + oximetria
4. Troponina (I ou T de alta sensibilidade)
5. Radiografia de tórax

Diagnósticos principais a excluir:
- IAMCSST: ST elevado ≥ 1mm em ≥ 2 derivações contíguas → cateterismo de emergência
- IAMSSST: ST deprimido, troponina elevada → estratificação de risco
- Dissecção aórtica: dor lancinante com irradiação dorsal → TC de aórta
- TEP: hipóxia, taquicardia, fator de risco → escore de Wells + D-dímero
- Pneumotórax: dispneia súbita, hipersonoridade → RX tórax

Escore HEART (IAMSSST/Angina instável):
H - História (altamente suspeita=2, moderada=1, pouco suspeita=0)
E - ECG (significativo=2, distúrbio inespecífico=1, normal=0)
A - Idade (≥65=2, 45-64=1, <45=0)
R - Fatores de risco (≥3 ou história de DAC=2, 1-2=1, nenhum=0)
T - Troponina (>3x normal=2, 1-3x=1, normal=0)
Score ≤ 3 = baixo risco (alta precoce), 4-6 = intermediário, ≥ 7 = alto risco
            """,
            "fonte": "Diretriz ACC/AHA 2021 + Protocolo Interno",
            "especialidade": "Cardiologia/Emergência"
        },
        {
            "conteudo": """
Protocolo de AVC Isquêmico Agudo
Fonte: AHA/ASA Stroke Guidelines 2019

Reconhecimento pré-hospitalar — SAMU:
- Sorriso assimétrico
- Abraço (fraqueza de membros)
- Mensagem / fala com dificuldade
- Urgência — ligar 192

Avaliação no hospital:
- TC de crânio sem contraste imediato (excluir hemorragia)
- Glicemia capilar imediata
- NIHSS (National Institutes of Health Stroke Scale)
- PA, ECG, oximetria

Trombólise IV (alteplase):
- Janela: até 4,5 horas do início dos sintomas
- Dose: 0,9 mg/kg (máximo 90 mg), 10% em bolus + 90% em 60 minutos
- Contraindicações absolutas: hemorragia intracraniana, cirurgia recente, PA > 185/110
- Monitorar PA a cada 15 minutos durante e após infusão

Trombectomia mecânica:
- Indicada em oclusão de grande vaso (ACI, ACM M1/M2)
- Janela: até 24 horas em pacientes selecionados
- Complementar ou substituta à trombólise

Cuidados gerais:
- PA: não tratar se < 220/120 (não candidatos à trombólise)
- Glicemia: manter 140-180 mg/dL
- Temperatura: tratar febres (AAS/paracetamol)
- Anticoagulação profilática para TEV
- Reabilitação precoce (fono, fisio, TO)
            """,
            "fonte": "AHA/ASA Stroke Guidelines 2019",
            "especialidade": "Neurologia"
        },
    ]

    documentos = []
    for conteudo_med in conteudos_medicos:
        documento = Document(
            page_content=conteudo_med["conteudo"],
            metadata={
                "fonte": conteudo_med["fonte"],
                "especialidade": conteudo_med["especialidade"],
            }
        )
        documentos.append(documento)

    return documentos


def carregar_documentos_do_diretorio(caminho_docs: str) -> list[Document]:
    """
    Carrega documentos de texto de um diretório para indexação no RAG.

    Parâmetros:
        caminho_docs: Caminho para o diretório com arquivos .txt

    Retorna:
        Lista de objetos Document carregados
    """
    caminho = Path(caminho_docs)

    if not caminho.exists() or not any(caminho.glob("*.txt")):
        logger.info(f"Nenhum documento encontrado em '{caminho_docs}'. Usando documentos de exemplo.")
        return criar_documentos_medicos_exemplo()

    try:
        loader = DirectoryLoader(
            caminho_docs,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documentos = loader.load()
        logger.info(f"Carregados {len(documentos)} documentos de '{caminho_docs}'")
        return documentos

    except Exception as erro:
        logger.warning(f"Erro ao carregar documentos: {erro}. Usando exemplos padrão.")
        return criar_documentos_medicos_exemplo()


def dividir_documentos(documentos: list[Document]) -> list[Document]:
    """
    Divide documentos longos em chunks menores para indexação eficiente.

    Parâmetros:
        documentos: Lista de documentos a dividir

    Retorna:
        Lista de chunks de documento
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_CHUNK,
        chunk_overlap=OVERLAP_CHUNK,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = divisor.split_documents(documentos)
    logger.info(f"Documentos divididos em {len(chunks)} chunks (tamanho: {TAMANHO_CHUNK})")
    return chunks


def criar_embeddings() -> HuggingFaceEmbeddings:
    """
    Inicializa o modelo de embeddings multilingual para vetorização.

    Usa o modelo paraphrase-multilingual-MiniLM que suporta português e inglês,
    essencial para indexar documentos em ambos os idiomas.

    Retorna:
        Objeto HuggingFaceEmbeddings configurado
    """
    logger.info(f"Carregando modelo de embeddings: {MODELO_EMBEDDINGS}")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings


def construir_ou_carregar_indice(
    caminho_indice: str = CAMINHO_INDICE_PADRAO,
    caminho_docs: str = CAMINHO_DOCS_PADRAO,
    forcar_reconstrucao: bool = False
) -> FAISS:
    """
    Constrói um novo índice FAISS ou carrega um existente do disco.

    Parâmetros:
        caminho_indice: Caminho para salvar/carregar o índice FAISS
        caminho_docs: Diretório com documentos para indexar
        forcar_reconstrucao: Se True, reconstrói mesmo que exista índice salvo

    Retorna:
        Índice FAISS pronto para consultas
    """
    embeddings = criar_embeddings()
    caminho = Path(caminho_indice)

    # Tenta carregar índice existente
    if caminho.exists() and not forcar_reconstrucao:
        logger.info(f"Carregando índice FAISS existente de: {caminho_indice}")
        try:
            indice = FAISS.load_local(
                caminho_indice,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Índice FAISS carregado com sucesso")
            return indice
        except Exception as erro:
            logger.warning(f"Falha ao carregar índice existente: {erro}. Reconstruindo...")

    # Constrói novo índice
    logger.info("Construindo novo índice FAISS...")
    documentos = carregar_documentos_do_diretorio(caminho_docs)
    chunks = dividir_documentos(documentos)

    indice = FAISS.from_documents(chunks, embeddings)

    # Salva o índice para uso futuro
    caminho.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(caminho))
    logger.info(f"Índice FAISS salvo em: {caminho_indice}")

    return indice


def criar_retriever(
    caminho_indice: str = CAMINHO_INDICE_PADRAO,
    caminho_docs: str = CAMINHO_DOCS_PADRAO,
    num_documentos: int = NUM_DOCUMENTOS_RECUPERADOS
):
    """
    Cria e retorna um retriever LangChain pronto para uso nas chains.

    Parâmetros:
        caminho_indice: Caminho para o índice FAISS
        caminho_docs: Diretório com documentos médicos
        num_documentos: Número de documentos a recuperar por consulta

    Retorna:
        Retriever LangChain configurado para buscas semânticas
    """
    logger.info("Inicializando retriever RAG...")

    indice = construir_ou_carregar_indice(
        caminho_indice=caminho_indice,
        caminho_docs=caminho_docs
    )

    retriever = indice.as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_documentos}
    )

    logger.info(f"Retriever RAG pronto (k={num_documentos} documentos por consulta)")
    return retriever


# Instância global do retriever (inicializada na primeira importação)
_retriever_global = None


def obter_retriever_global():
    """
    Retorna a instância global do retriever, criando se necessário.

    Retorna:
        Retriever RAG compartilhado entre todas as chains
    """
    global _retriever_global

    if _retriever_global is None:
        caminho_indice = os.environ.get("FAISS_INDEX_PATH", CAMINHO_INDICE_PADRAO)
        caminho_docs = os.environ.get("DOCS_PATH", CAMINHO_DOCS_PADRAO)
        _retriever_global = criar_retriever(caminho_indice, caminho_docs)

    return _retriever_global


def buscar_com_scores(query: str, num_documentos: int = NUM_DOCUMENTOS_RECUPERADOS) -> list[dict]:
    """
    Executa busca semântica retornando documentos com scores de relevância (0-1).

    Parâmetros:
        query: Texto da consulta
        num_documentos: Número de documentos a recuperar

    Retorna:
        Lista de dicts com posicao_ranking, conteudo, fonte e score_similaridade
    """
    retriever = obter_retriever_global()
    indice = retriever.vectorstore
    pares = indice.similarity_search_with_relevance_scores(query, k=num_documentos)

    resultados = []
    for posicao, (doc, score) in enumerate(pares):
        resultados.append({
            "posicao_ranking": posicao + 1,
            "conteudo": doc.page_content[:300],
            "fonte": doc.metadata.get("fonte", "Protocolo interno"),
            "score_similaridade": round(float(score), 4),
        })

    return resultados
