"""
logger.py
---------
Registra todas as interações do assistente médico para fins de auditoria.

Cada entrada de log contém:
- Timestamp ISO 8601
- ID único da sessão
- Pergunta do usuário (anonimizada)
- Resposta gerada pelo assistente
- Chain LangChain utilizada
- Fontes e protocolos citados
- Flag de alerta (quando safety.py interveio)

Os logs são salvos em formato JSONL (uma linha JSON por interação)
para facilitar análise posterior com ferramentas de processamento de dados.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ============================================================
# CONSTANTES
# ============================================================
CAMINHO_LOG_PADRAO = "logs/interactions.jsonl"
NIVEL_LOG_PADRAO = "INFO"
FORMATO_TIMESTAMP = "%Y-%m-%dT%H:%M:%S.%f%z"

# Configura o logger de sistema
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger_sistema = logging.getLogger(__name__)


class AuditoriaLogger:
    """
    Logger de auditoria para todas as interações do assistente médico.

    Responsável por registrar de forma estruturada cada conversa,
    anonimizando dados sensíveis antes de persistir em disco.

    Atributos:
        caminho_log: Caminho do arquivo JSONL de logs
        id_sessao: Identificador único da sessão atual
    """

    def __init__(self, caminho_log: str = CAMINHO_LOG_PADRAO) -> None:
        """
        Inicializa o logger de auditoria.

        Parâmetros:
            caminho_log: Caminho para o arquivo de log JSONL
        """
        self.caminho_log = Path(caminho_log)
        self.id_sessao = str(uuid.uuid4())[:8]

        # Garante que o diretório de logs existe
        self.caminho_log.parent.mkdir(parents=True, exist_ok=True)

        logger_sistema.info(
            f"Logger de auditoria iniciado. "
            f"Sessão: {self.id_sessao} | Arquivo: {self.caminho_log}"
        )

    def _obter_timestamp(self) -> str:
        """
        Retorna o timestamp atual no formato ISO 8601 com timezone UTC.

        Retorna:
            String com timestamp formatado
        """
        agora = datetime.now(timezone.utc)
        return agora.strftime(FORMATO_TIMESTAMP)

    def _anonimizar_texto(self, texto: str) -> str:
        """
        Remove informações pessoais identificáveis do texto antes de logar.

        Aplica as mesmas transformações do prepare_dataset.py para garantir
        consistência na anonimização de toda a plataforma.

        Parâmetros:
            texto: Texto potencialmente contendo dados sensíveis

        Retorna:
            Texto com dados sensíveis substituídos por marcadores
        """
        if not isinstance(texto, str):
            return str(texto)

        # Remove referências a pacientes pelo nome (suporta letras acentuadas)
        texto = re.sub(
            r'\b(?:patient|paciente|pt\.?)\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*\b',
            '[PACIENTE]',
            texto,
            flags=re.IGNORECASE
        )

        # Remove datas
        texto = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATA]', texto)

        # Remove números de registro médico
        texto = re.sub(
            r'\b(?:MRN|registro|ID)\s*[:#]?\s*\d{4,10}\b',
            '[ID_REGISTRO]',
            texto,
            flags=re.IGNORECASE
        )

        # Remove CPF
        texto = re.sub(r'\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{2}\b', '[CPF]', texto)

        # Remove números de telefone
        texto = re.sub(
            r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{4,5}[-.\s]?\d{4}\b',
            '[TELEFONE]',
            texto
        )

        # Remove endereços de e-mail
        texto = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            texto
        )

        return texto

    def registrar_interacao(
        self,
        pergunta_usuario: str,
        resposta_assistente: str,
        chain_utilizada: str,
        fontes_citadas: Optional[list[str]] = None,
        flag_alerta: bool = False,
        metadata_adicional: Optional[dict] = None,
        documentos_rag: Optional[list[dict]] = None,
        confianca_estimada: Optional[float] = None,
        motivo_resposta: Optional[str] = None,
    ) -> str:
        """
        Registra uma interação completa no arquivo de auditoria.

        Parâmetros:
            pergunta_usuario: Texto da pergunta feita pelo usuário
            resposta_assistente: Resposta gerada pelo assistente
            chain_utilizada: Nome da chain LangChain que processou a requisição
            fontes_citadas: Lista de protocolos e fontes mencionados na resposta
            flag_alerta: True se o módulo de segurança interveio na resposta
            metadata_adicional: Dados extras opcionais para debug

        Retorna:
            ID único da interação registrada
        """
        id_interacao = str(uuid.uuid4())[:12]

        # Calcula confiança estimada se não fornecida explicitamente
        if confianca_estimada is None and documentos_rag:
            scores = [d.get("score_similaridade", 0) for d in documentos_rag]
            base = sum(scores) / len(scores) if scores else 0.0
            confianca_estimada = round(base * (0.8 if flag_alerta else 1.0), 4)

        entrada_log = {
            "id_interacao": id_interacao,
            "id_sessao": self.id_sessao,
            "timestamp": self._obter_timestamp(),
            "chain_utilizada": chain_utilizada,
            "pergunta_anonimizada": self._anonimizar_texto(pergunta_usuario),
            "resposta_gerada": resposta_assistente,
            "comprimento_pergunta": len(pergunta_usuario),
            "comprimento_resposta": len(resposta_assistente),
            "fontes_citadas": fontes_citadas or [],
            "flag_alerta_seguranca": flag_alerta,
            "documentos_rag": documentos_rag or [],
            "confianca_estimada": confianca_estimada,
            "motivo_resposta": motivo_resposta,
            "metadata": metadata_adicional or {},
        }

        self._persistir_log(entrada_log)
        logger_sistema.info(
            f"[Sessão:{self.id_sessao}] Interação {id_interacao} registrada. "
            f"Chain: {chain_utilizada} | Alerta: {flag_alerta}"
        )

        return id_interacao

    def _persistir_log(self, entrada: dict) -> None:
        """
        Persiste a entrada de log no arquivo JSONL em modo append.

        Parâmetros:
            entrada: Dicionário com os dados da interação
        """
        try:
            with open(self.caminho_log, "a", encoding="utf-8") as arquivo:
                arquivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        except IOError as erro:
            logger_sistema.error(f"Erro ao persistir log: {erro}")

    def recuperar_interacoes_recentes(self, num_interacoes: int = 20) -> list[dict]:
        """
        Recupera as interações mais recentes do arquivo de log.

        Parâmetros:
            num_interacoes: Número máximo de interações a recuperar

        Retorna:
            Lista de dicionários com as interações mais recentes
        """
        if not self.caminho_log.exists():
            logger_sistema.warning("Arquivo de log não encontrado")
            return []

        try:
            with open(self.caminho_log, "r", encoding="utf-8") as arquivo:
                linhas = arquivo.readlines()

            # Pega as últimas N linhas
            linhas_recentes = linhas[-num_interacoes:]
            interacoes = []

            for linha in linhas_recentes:
                linha = linha.strip()
                if linha:
                    try:
                        interacoes.append(json.loads(linha))
                    except json.JSONDecodeError:
                        continue

            return interacoes

        except IOError as erro:
            logger_sistema.error(f"Erro ao ler logs: {erro}")
            return []

    def contar_interacoes_sessao(self) -> int:
        """
        Conta o número de interações na sessão atual.

        Retorna:
            Número de interações registradas na sessão corrente
        """
        interacoes = self.recuperar_interacoes_recentes(num_interacoes=1000)
        return sum(1 for i in interacoes if i.get("id_sessao") == self.id_sessao)

    def obter_estatisticas(self) -> dict:
        """
        Calcula estatísticas gerais dos logs de auditoria.

        Retorna:
            Dicionário com métricas de uso do sistema
        """
        todas_interacoes = self.recuperar_interacoes_recentes(num_interacoes=10000)

        if not todas_interacoes:
            return {"total_interacoes": 0, "total_alertas": 0}

        total_alertas = sum(
            1 for i in todas_interacoes if i.get("flag_alerta_seguranca")
        )

        chains_usadas = {}
        for interacao in todas_interacoes:
            chain = interacao.get("chain_utilizada", "desconhecida")
            chains_usadas[chain] = chains_usadas.get(chain, 0) + 1

        return {
            "total_interacoes": len(todas_interacoes),
            "total_alertas": total_alertas,
            "percentual_alertas": round(total_alertas / len(todas_interacoes) * 100, 2),
            "chains_mais_usadas": sorted(
                chains_usadas.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }


# Instância global do logger para uso em toda a aplicação
logger_auditoria = AuditoriaLogger(
    caminho_log=os.environ.get("LOG_PATH", CAMINHO_LOG_PADRAO)
)
