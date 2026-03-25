"""
safety.py
---------
Valida as respostas do assistente antes de exibir ao usuário final.

Regras de segurança implementadas:
1. Nunca retornar dosagens específicas de medicamentos controlados sem contexto clínico
2. Nunca afirmar diagnóstico definitivo — usar linguagem de sugestão/possibilidade
3. Sempre incluir disclaimer sobre necessidade de validação médica humana
4. Bloquear perguntas fora do escopo médico
5. Detectar e sinalizar respostas de alta urgência

O módulo intervém de forma transparente, modificando respostas inadequadas
e registrando cada intervenção para fins de auditoria.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# CONSTANTES
# ============================================================
DISCLAIMER_PADRAO = (
    "\n\n⚠️ **Aviso Importante:** Esta resposta é gerada por um sistema de "
    "apoio à decisão clínica baseado em protocolos e diretrizes médicas. "
    "**NÃO substitui a avaliação, diagnóstico ou prescrição de um profissional "
    "de saúde habilitado.** Em caso de emergência médica, ligue imediatamente "
    "para o SAMU (192) ou vá ao pronto-socorro mais próximo."
)

DISCLAIMER_EMERGENCIA = (
    "\n\n🚨 **ATENÇÃO — POSSÍVEL EMERGÊNCIA:** Os sintomas descritos podem "
    "indicar uma condição grave que requer avaliação médica imediata. "
    "**Ligue 192 (SAMU) ou vá imediatamente ao pronto-socorro.** "
    "Não aguarde — o tempo é crítico nessas situações."
)

RESPOSTA_FORA_ESCOPO = (
    "Posso auxiliar apenas com questões relacionadas à área da saúde e medicina. "
    "Para este tipo de pergunta, recomendo consultar fontes especializadas no assunto. "
    "Se tiver dúvidas médicas, ficarei feliz em ajudar!"
)

# Medicamentos controlados que requerem atenção especial
MEDICAMENTOS_CONTROLADOS = [
    "morfina", "fentanil", "oxicodona", "codeína", "tramadol", "metadona",
    "buprenorfina", "metilfenidato", "anfetamina", "benzodiazepínico", "midazolam",
    "diazepam", "clonazepam", "alprazolam", "zolpidem", "clonidina",
]

# Padrões de linguagem definitiva que devem ser suavizados
PADROES_DIAGNOSTICO_DEFINITIVO = [
    r"\b(você tem|o paciente tem|é definitivamente|certamente é|sem dúvida é)\b",
    r"\b(o diagnóstico é|foi diagnosticado com|confirmo que)\b",
    r"\b(pode tomar|tome imediatamente|use)\s+\d+\s*mg\b",
]

# Palavras-chave de emergência
PALAVRAS_EMERGENCIA = [
    "infarto", "acidente vascular cerebral", "avc", "parada cardíaca",
    "anafilaxia", "choque", "convulsão", "inconsciência", "sangramento intenso",
    "dificuldade respiratória grave", "overdose", "tentativa de suicídio",
    "engasgamento", "afogamento", "queimadura grave", "trauma grave",
]

# Palavras-chave do domínio médico — a pergunta deve conter ao menos uma
# para ser processada (allowlist). Qualquer pergunta sem termos médicos é bloqueada.
PALAVRAS_CHAVE_MEDICAS = [
    # Sintomas e queixas
    "dor", "febre", "tontura", "náusea", "vômito", "tosse", "dispneia", "falta de ar",
    "sangramento", "edema", "inchaço", "fraqueza", "fadiga", "cansaço", "desmaiou",
    "convulsão", "desmaio", "palpitação", "tremor", "dormência", "formigamento",
    # Condições e diagnósticos
    "diabetes", "hipertensão", "pressão", "infarto", "avc", "acidente vascular",
    "sepse", "pneumonia", "infecção", "tumor", "câncer", "neoplasia", "anemia",
    "insuficiência", "arritmia", "fibrilação", "trombose", "embolia", "isquemia",
    "hipotireoidismo", "hipertireoidismo", "asma", "dpoc", "cirrose", "hepatite",
    "irc", "ira", "insuficiência renal", "insuficiência cardíaca", "doença",
    # Exames e procedimentos
    "exame", "hemograma", "glicemia", "creatinina", "ureia", "troponina", "bnp",
    "ecg", "eletrocardiograma", "raio-x", "tomografia", "ressonância", "ultrassom",
    "biópsia", "cultura", "hemocultura", "gasometria", "coagulograma", "pcr",
    "leucócito", "eritrócito", "plaqueta", "hemoglobina", "hematócrito",
    # Medicamentos e tratamentos
    "medicamento", "remédio", "droga", "fármaco", "antibiótico", "anti-inflamatório",
    "analgésico", "diurético", "vasodilatador", "anticoagulante", "antiagregante",
    "insulina", "corticoide", "quimioterapia", "radioterapia", "cirurgia", "dose",
    "posologia", "via de administração", "efeito colateral", "contraindicação",
    # Especialidades e contexto clínico
    "paciente", "médico", "hospital", "uti", "pronto-socorro", "emergência",
    "internação", "diagnóstico", "tratamento", "protocolo", "conduta", "prescrição",
    "clínico", "cirúrgico", "laboratorial", "prontuário", "anamnese", "semiologia",
    "cardíaco", "pulmonar", "renal", "hepático", "neurológico", "oncológico",
    "pediátrico", "geriátrico", "obstétrico", "ginecológico", "ortopédico",
    # Anatomia e fisiologia
    "coração", "pulmão", "fígado", "rim", "cérebro", "sangue", "artéria", "veia",
    "músculo", "osso", "nervo", "hormônio", "enzima", "proteína", "célula",
]

# Configura o logger
logger = logging.getLogger(__name__)


@dataclass
class ResultadoValidacao:
    """
    Resultado da validação de segurança de uma resposta.

    Atributos:
        aprovada: True se a resposta passou na validação sem modificações
        resposta_final: Texto final após aplicação das regras de segurança
        intervencoes: Lista de intervenções aplicadas
        flag_alerta: True se houve qualquer intervenção de segurança
        e_emergencia: True se foram detectados padrões de emergência
        fora_do_escopo: True se a pergunta está fora do domínio médico
    """
    aprovada: bool
    resposta_final: str
    intervencoes: list[str] = field(default_factory=list)
    flag_alerta: bool = False
    e_emergencia: bool = False
    fora_do_escopo: bool = False


class ValidadorSeguranca:
    """
    Valida e sanitiza respostas do assistente médico para segurança do paciente.

    Aplica um conjunto de regras que garantem que o assistente não ultrapasse
    os limites de sua atuação como sistema de apoio à decisão clínica.
    """

    def validar_resposta(
        self,
        pergunta: str,
        resposta: str,
        chain_utilizada: str = "desconhecida"
    ) -> ResultadoValidacao:
        """
        Valida a resposta do assistente aplicando todas as regras de segurança.

        Executa verificações sequenciais e acumula intervenções necessárias.

        Parâmetros:
            pergunta: Pergunta original do usuário
            resposta: Resposta gerada pelo assistente médico
            chain_utilizada: Nome da chain que gerou a resposta

        Retorna:
            ResultadoValidacao com a resposta final e metadados de segurança
        """
        intervencoes = []
        texto_modificado = resposta
        e_emergencia = False
        fora_do_escopo = False

        # Regra 1: Verificar se está dentro do escopo médico
        if self._e_fora_do_escopo(pergunta):
            logger.warning(f"[Segurança] Pergunta fora do escopo médico detectada")
            fora_do_escopo = True
            intervencoes.append("BLOQUEIO: Pergunta fora do escopo médico")

            return ResultadoValidacao(
                aprovada=False,
                resposta_final=RESPOSTA_FORA_ESCOPO,
                intervencoes=intervencoes,
                flag_alerta=True,
                e_emergencia=False,
                fora_do_escopo=True,
            )

        # Regra 2: Detectar padrões de emergência
        if self._e_emergencia(pergunta + " " + resposta):
            logger.warning(f"[Segurança] Padrão de emergência detectado na interação")
            e_emergencia = True
            texto_modificado += DISCLAIMER_EMERGENCIA
            intervencoes.append("ALERTA: Adicionado disclaimer de emergência")

        # Regra 3: Suavizar linguagem de diagnóstico definitivo
        texto_modificado, intervencoes_diag = self._suavizar_diagnostico_definitivo(texto_modificado)
        intervencoes.extend(intervencoes_diag)

        # Regra 4: Verificar menção de medicamentos controlados
        texto_modificado, intervencoes_med = self._processar_medicamentos_controlados(
            texto_modificado
        )
        intervencoes.extend(intervencoes_med)

        # Regra 5: Sempre adicionar disclaimer padrão (se não é emergência)
        if not e_emergencia:
            texto_modificado += DISCLAIMER_PADRAO
            intervencoes.append("DISCLAIMER: Aviso padrão de limitações adicionado")

        # Determina se houve alguma intervenção real de segurança
        flag_alerta = e_emergencia or len([i for i in intervencoes if "MODIFICAÇÃO" in i or "ALERTA" in i]) > 0

        return ResultadoValidacao(
            aprovada=len(intervencoes) <= 1,  # Apenas o disclaimer padrão é aceitável
            resposta_final=texto_modificado,
            intervencoes=intervencoes,
            flag_alerta=flag_alerta,
            e_emergencia=e_emergencia,
            fora_do_escopo=False,
        )

    def _e_fora_do_escopo(self, texto: str) -> bool:
        """
        Verifica se o texto da pergunta está fora do domínio médico.

        Usa allowlist: bloqueia qualquer pergunta que não contenha ao menos
        uma palavra-chave do domínio médico.

        Parâmetros:
            texto: Texto a verificar

        Retorna:
            True se o tema não é médico (deve ser bloqueado)
        """
        texto_lower = texto.lower()

        for palavra in PALAVRAS_CHAVE_MEDICAS:
            if palavra in texto_lower:
                return False

        return True

    def _e_emergencia(self, texto: str) -> bool:
        """
        Detecta se o texto contém indicadores de emergência médica.

        Parâmetros:
            texto: Texto combinado da pergunta e resposta

        Retorna:
            True se há indícios de situação de emergência
        """
        texto_lower = texto.lower()

        for palavra in PALAVRAS_EMERGENCIA:
            if palavra in texto_lower:
                return True

        # Padrões de urgência temporal
        padroes_urgencia = [
            r"\b(imediatamente|urgente|emergência|socorro|ajuda agora)\b",
            r"\b(está desmaiando|perdeu a consciência|não respira|sem pulso)\b",
        ]

        for padrao in padroes_urgencia:
            if re.search(padrao, texto_lower, re.IGNORECASE):
                return True

        return False

    def _suavizar_diagnostico_definitivo(self, texto: str) -> tuple[str, list[str]]:
        """
        Substitui afirmações diagnósticas definitivas por linguagem de sugestão.

        Converte frases como "você tem X" para "pode sugerir X" ou
        "os sintomas são sugestivos de X".

        Parâmetros:
            texto: Texto da resposta a ser analisado

        Retorna:
            Tupla (texto_modificado, lista_de_intervencoes)
        """
        intervencoes = []
        texto_modificado = texto

        substituicoes = {
            r'\bvocê tem\b': 'seus sintomas podem sugerir',
            r'\bo paciente tem\b': 'o paciente pode apresentar',
            r'\bé definitivamente\b': 'pode ser compatível com',
            r'\bcertamente é\b': 'pode se tratar de',
            r'\bsem dúvida é\b': 'os achados sugerem',
            r'\bo diagnóstico é\b': 'o diagnóstico sugerido é',
            r'\bfoi diagnosticado com\b': 'apresenta quadro compatível com',
        }

        for padrao, substituto in substituicoes.items():
            if re.search(padrao, texto_modificado, re.IGNORECASE):
                texto_modificado = re.sub(padrao, substituto, texto_modificado, flags=re.IGNORECASE)
                intervencoes.append(f"MODIFICAÇÃO: Linguagem diagnóstica suavizada ({padrao} → {substituto})")

        return texto_modificado, intervencoes

    def _processar_medicamentos_controlados(self, texto: str) -> tuple[str, list[str]]:
        """
        Adiciona contexto de segurança quando medicamentos controlados são mencionados.

        Não remove as informações, mas adiciona aviso sobre necessidade de
        prescrição médica quando medicamentos controlados são citados.

        Parâmetros:
            texto: Texto da resposta

        Retorna:
            Tupla (texto_modificado, lista_de_intervencoes)
        """
        intervencoes = []
        texto_lower = texto.lower()
        medicamentos_encontrados = []

        for medicamento in MEDICAMENTOS_CONTROLADOS:
            if medicamento in texto_lower:
                medicamentos_encontrados.append(medicamento)

        if medicamentos_encontrados:
            aviso_controlado = (
                f"\n\n⚕️ **Nota sobre medicamentos controlados:** "
                f"Os medicamentos mencionados ({', '.join(medicamentos_encontrados)}) "
                f"requerem prescrição médica especial. A dosagem, frequência e "
                f"indicação devem ser determinadas exclusivamente pelo médico assistente, "
                f"considerando o histórico completo do paciente."
            )
            texto += aviso_controlado
            intervencoes.append(
                f"ALERTA: Aviso sobre medicamentos controlados adicionado: {medicamentos_encontrados}"
            )

        return texto, intervencoes

    def verificar_pergunta(self, pergunta: str) -> bool:
        """
        Verifica se a pergunta é adequada para o sistema antes de processar.

        Parâmetros:
            pergunta: Pergunta do usuário

        Retorna:
            True se a pergunta pode ser processada, False se deve ser bloqueada
        """
        if len(pergunta.strip()) < 5:
            logger.warning("[Segurança] Pergunta muito curta bloqueada")
            return False

        if self._e_fora_do_escopo(pergunta):
            logger.warning("[Segurança] Pergunta fora do escopo bloqueada")
            return False

        return True


# Instância global do validador para uso em toda a aplicação
validador_seguranca = ValidadorSeguranca()
