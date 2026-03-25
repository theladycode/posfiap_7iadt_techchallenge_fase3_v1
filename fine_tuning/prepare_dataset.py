"""
prepare_dataset.py
------------------
Responsável por:
1. Baixar os datasets PubMedQA e MedQuAD
2. Limpar e padronizar os dados
3. Anonimizar informações sensíveis
4. Gerar dados sintéticos complementares em português
5. Salvar no formato de instrução para fine-tuning

Formato de saída de cada exemplo:
{
    "instruction": "Você é um assistente médico. Responda a pergunta clínica:",
    "input": "Qual o tratamento recomendado para hipertensão estágio 1?",
    "output": "O tratamento de primeira linha inclui mudanças no estilo de vida..."
}
"""

import json
import re
import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

# ============================================================
# CONSTANTES
# ============================================================
DATASET_PUBMEDQA = "qiaojin/PubMedQA"
CONFIG_PUBMEDQA = "pqa_labeled"
INSTRUCAO_SISTEMA = (
    "Você é um assistente médico de apoio clínico. "
    "Responda a pergunta clínica com base em evidências científicas:"
)
CAMINHO_SAIDA = Path("data/processed/medical_dataset.json")
CAMINHO_SINTETICOS = Path("data/synthetic/synthetic_examples.json")
NUM_EXEMPLOS_SINTETICOS = 50
SEMENTE_ALEATORIA = 42

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================
# EXEMPLOS SINTÉTICOS EM PORTUGUÊS
# ============================================================
PARES_SINTETICOS = [
    {
        "pergunta": "Qual o tratamento de primeira linha para hipertensão arterial estágio 1?",
        "resposta": (
            "Para hipertensão estágio 1 (PAS 130-139 ou PAD 80-89 mmHg), o tratamento "
            "de primeira linha inclui modificações no estilo de vida: redução do consumo "
            "de sal para menos de 5g/dia, prática de atividade física aeróbica moderada "
            "por pelo menos 150 minutos semanais, redução do peso corporal, cessação do "
            "tabagismo e moderação no consumo de álcool. Farmacoterapia pode ser iniciada "
            "em pacientes com risco cardiovascular elevado. Fonte: Diretriz Brasileira de "
            "Hipertensão Arterial 2020."
        )
    },
    {
        "pergunta": "Quais são os critérios diagnósticos para diabetes mellitus tipo 2?",
        "resposta": (
            "O diagnóstico de diabetes mellitus tipo 2 pode ser feito por qualquer um dos "
            "critérios: (1) glicemia de jejum >= 126 mg/dL em duas ocasiões distintas; "
            "(2) glicemia 2h após sobrecarga com 75g de glicose >= 200 mg/dL; (3) HbA1c "
            ">= 6,5% em metodologia padronizada; (4) glicemia aleatória >= 200 mg/dL na "
            "presença de sintomas clássicos (poliúria, polidipsia, perda de peso). "
            "Fonte: Diretrizes da Sociedade Brasileira de Diabetes 2023."
        )
    },
    {
        "pergunta": "Como calcular o escore de Wells para trombose venosa profunda?",
        "resposta": (
            "O escore de Wells para TVP atribui pontos para: câncer ativo (+1), paralisia "
            "ou imobilização de MMII (+1), acamado > 3 dias ou cirurgia nas últimas 4 "
            "semanas (+1), dor localizada em trajeto venoso profundo (+1), edema de toda "
            "a perna (+1), edema com cacifo apenas na perna sintomática (+1), veias "
            "superficiais colaterais (não varicosas) (+1), TVP prévia (+1), diagnóstico "
            "alternativo tão provável ou mais (-2). Pontuação: >= 2 = alta probabilidade; "
            "0-1 = baixa probabilidade. Fonte: Wells PS et al., Lancet 1997."
        )
    },
    {
        "pergunta": "Quais são os sinais de alarme na cefaleia que indicam investigação urgente?",
        "resposta": (
            "Os sinais de alarme ('red flags') na cefaleia que indicam investigação urgente "
            "incluem: início súbito e de forte intensidade ('thunderclap headache'), mudança "
            "no padrão habitual da cefaleia, cefaleia progressiva e de piora, associação com "
            "febre e rigidez de nuca, déficit neurológico focal, papiledema, início após os "
            "50 anos, cefaleia desencadeada por manobra de Valsalva, cefaleia em "
            "imunossuprimidos ou com câncer. Esses casos requerem neuroimagem e avaliação "
            "médica imediata. Fonte: International Headache Society 2018."
        )
    },
    {
        "pergunta": "Qual a conduta inicial no infarto agudo do miocárdio com supradesnivelamento de ST?",
        "resposta": (
            "A conduta inicial no IAMCSST inclui: (1) acesso venoso e monitorização cardíaca "
            "contínua; (2) oxigênio se SatO2 < 90%; (3) AAS 300mg mastigado + inibidor de "
            "P2Y12 (ticagrelor 180mg ou clopidogrel 600mg); (4) anticoagulação (heparina não "
            "fracionada ou enoxaparina); (5) reperfusão: ICP primária em até 90 minutos do "
            "primeiro contato médico (preferencial) ou fibrinólise se ICP indisponível em "
            "até 120 minutos. O tempo é determinante para preservação miocárdica. "
            "Fonte: Diretriz AHA/ACC 2022."
        )
    },
    {
        "pergunta": "Como diferenciar pneumonia viral de bacteriana na prática clínica?",
        "resposta": (
            "A diferenciação entre pneumonia viral e bacteriana é desafiadora clinicamente. "
            "Sugestivo de pneumonia bacteriana: início agudo, febre alta (> 38,5°C), calafrios, "
            "expectoração purulenta, consolidação lobar na radiografia, leucocitose com "
            "desvio à esquerda, PCR e procalcitonina elevadas. Sugestivo de pneumonia viral: "
            "pródromo de via aérea superior, febre moderada, tosse seca, infiltrado intersticial "
            "bilateral, leucócitos normais ou linfocitose. A sobreposição é comum e o contexto "
            "epidemiológico (pandemia, surto) é relevante. Fonte: IDSA Guidelines 2019."
        )
    },
    {
        "pergunta": "Quais são as indicações de internação hospitalar na pancreatite aguda?",
        "resposta": (
            "As indicações de internação na pancreatite aguda incluem: (1) dor intratável "
            "ambulatorialmente; (2) vômitos com incapacidade de hidratação oral; (3) pancreatite "
            "moderada ou grave pelo escore de Atlanta revisado (falência orgânica transitória ou "
            "persistente); (4) escore de Ranson >= 3 ou APACHE II >= 8; (5) complicações locais "
            "(pseudocisto, necrose); (6) comorbidades descompensadas. Todos os casos devem ter "
            "hidratação vigorosa iniciada nas primeiras 24h. Fonte: ACG Guidelines 2013."
        )
    },
    {
        "pergunta": "Como manejar a hipoglicemia grave em paciente diabético?",
        "resposta": (
            "Para hipoglicemia grave (paciente inconsciente ou incapaz de deglutir): (1) "
            "administrar glucagon 1mg IM ou SC se disponível; (2) acesso venoso e glicose "
            "50% IV: 20-50mL em adultos (0,5-1g/kg em crianças); (3) monitorar glicemia a "
            "cada 15 minutos; (4) após recuperação da consciência, oferecer carboidratos "
            "complexos por via oral; (5) investigar causa (dose excessiva, jejum prolongado, "
            "exercício, interação medicamentosa). Prevenir com ajuste da terapia antidiabética "
            "e educação do paciente e familiar. Fonte: ADA Standards of Care 2024."
        )
    },
    {
        "pergunta": "Quais são as contraindicações absolutas ao uso de trombolíticos no AVC isquêmico?",
        "resposta": (
            "As contraindicações absolutas ao alteplase no AVC isquêmico incluem: (1) TC com "
            "hemorragia intracraniana; (2) cirurgia intracraniana/espinal ou TCE grave nos "
            "últimos 3 meses; (3) histórico de AVC e DM concomitante; (4) PA sistólica > "
            "185mmHg ou diastólica > 110mmHg irresponsiva ao tratamento; (5) glicemia < "
            "50 ou > 400mg/dL; (6) uso de anticoagulantes orais; (7) contagem de plaquetas "
            "< 100.000/mm³; (8) cirurgia de grande porte nos últimos 14 dias; (9) sangramento "
            "ativo. A janela terapêutica padrão é de até 4,5 horas do início dos sintomas. "
            "Fonte: AHA/ASA Guidelines 2019."
        )
    },
    {
        "pergunta": "Como avaliar a gravidade da insuficiência cardíaca pela classificação NYHA?",
        "resposta": (
            "A classificação funcional da NYHA (New York Heart Association) estratifica a "
            "insuficiência cardíaca em 4 classes: Classe I - sem limitação da atividade física "
            "habitual, sem sintomas; Classe II - leve limitação, sintomas (dispneia, fadiga) "
            "aos grandes esforços, confortável em repouso; Classe III - importante limitação, "
            "sintomas aos pequenos esforços, confortável em repouso; Classe IV - incapacidade "
            "de realizar qualquer atividade sem desconforto, sintomas em repouso. A classificação "
            "guia decisões terapêuticas e prognóstico. Fonte: AHA/ACC Heart Failure Guidelines 2022."
        )
    },
    {
        "pergunta": "Qual a abordagem diagnóstica para anemia ferropriva?",
        "resposta": (
            "A investigação da anemia ferropriva inclui: (1) hemograma: anemia microcítica e "
            "hipocrômica, trombocitose reativa; (2) ferritina sérica < 30 ng/mL (principal "
            "marcador de depleção de ferro); (3) ferro sérico reduzido e TIBC aumentado, "
            "saturação de transferrina < 16%; (4) investigação da causa: sangramento oculto "
            "(pesquisa de sangue oculto nas fezes, endoscopia), má absorção (doença celíaca, "
            "gastrectomia), demanda aumentada (gravidez). O tratamento é sulfato ferroso 200mg "
            "VO 2-3x/dia em jejum. Fonte: WHO Iron Deficiency Anaemia Guidelines."
        )
    },
    {
        "pergunta": "Como diagnosticar e tratar a otite média aguda?",
        "resposta": (
            "O diagnóstico de otite média aguda requer: (1) início agudo dos sintomas; (2) "
            "derrame no ouvido médio (membrana abaulada, opacificada, sem mobilidade); (3) "
            "sinais de inflamação (eritema, otalgia). Tratamento: em crianças < 2 anos e casos "
            "graves: amoxicilina 80-90mg/kg/dia por 10 dias; em crianças >= 2 anos com sintomas "
            "leves-moderados: pode-se adotar conduta expectante por 48-72h com analgesia. "
            "Falha terapêutica: amoxicilina-clavulanato. Alergia a penicilina: azitromicina. "
            "Fonte: AAP Clinical Practice Guideline 2013, atualizada 2022."
        )
    },
    {
        "pergunta": "Quais são os critérios de SIRS e sepse?",
        "resposta": (
            "SIRS (Síndrome da Resposta Inflamatória Sistêmica) - 2 ou mais: temperatura > "
            "38°C ou < 36°C; FC > 90bpm; FR > 20irpm ou PaCO2 < 32mmHg; leucócitos > "
            "12.000 ou < 4.000/mm³ ou > 10% bastões. Sepse (Sepse-3, 2016): disfunção "
            "orgânica ameaçadora à vida causada por resposta desregulada do hospedeiro à "
            "infecção - aumento de 2 pontos no escore SOFA. Choque séptico: sepse + necessidade "
            "de vasopressor para PAM >= 65mmHg + lactato > 2 mmol/L apesar de ressuscitação "
            "volêmica adequada. Fonte: Surviving Sepsis Campaign 2021."
        )
    },
    {
        "pergunta": "Como manejar a crise asmática grave no pronto-socorro?",
        "resposta": (
            "Na crise asmática grave: (1) oxigênio para manter SatO2 >= 93%; (2) salbutamol "
            "2,5-5mg em nebulização a cada 20 minutos na primeira hora (ou 4-8 puffs com "
            "espaçador); (3) brometo de ipratrópio 0,5mg associado ao salbutamol nas primeiras "
            "horas; (4) corticosteroide sistêmico: prednisolona 40-60mg VO ou metilprednisolona "
            "1-2mg/kg IV; (5) sulfato de magnésio 2g IV em 20 minutos nos casos graves; "
            "(6) considerar ventilação não-invasiva ou IOT se deterioração. Monitorar PFE e "
            "SpO2 continuamente. Fonte: GINA 2023."
        )
    },
    {
        "pergunta": "Quais são os fatores de risco para tromboembolismo venoso?",
        "resposta": (
            "Os fatores de risco para TEV dividem-se em: Maiores: cirurgia ortopédica de "
            "MMII, fraturas de quadril/perna, imobilização prolongada, neoplasia ativa, "
            "TEV prévio, trombofilia hereditária. Moderados: cirurgia geral, insuficiência "
            "cardíaca ou respiratória, gravidez/puerpério, contraceptivos orais, reposição "
            "hormonal, doença inflamatória intestinal. Menores: obesidade, viagem prolongada, "
            "varizes, tabagismo. A profilaxia com heparina de baixo peso molecular é indicada "
            "em pacientes hospitalizados com risco moderado-alto. Fonte: ACCP Guidelines 2016."
        )
    },
    {
        "pergunta": "Como interpretar um eletrocardiograma com bloqueio de ramo esquerdo?",
        "resposta": (
            "O bloqueio de ramo esquerdo (BRE) completo no ECG apresenta: (1) QRS >= 120ms; "
            "(2) ondas R largas e entalhadas (padrão RR' ou 'M') em V5-V6, D1 e aVL; (3) "
            "ondas S profundas em V1-V3; (4) desvio do eixo para esquerda; (5) discordância "
            "ST-T (alterações secundárias). O BRE novo ou presumivelmente novo em contexto "
            "clínico de dor torácica é equivalente a IAMCSST e indica reperfusão urgente "
            "(critério de Sgarbossa pode auxiliar). BRE crônico requer investigação de "
            "cardiopatia estrutural. Fonte: AHA/ACC 2009."
        )
    },
    {
        "pergunta": "Qual a conduta na hipercalemia grave com alterações eletrocardiográficas?",
        "resposta": (
            "Na hipercalemia grave (K+ > 6,5 mEq/L ou com alterações no ECG): (1) gluconato "
            "de cálcio 10% 10-20mL IV em 10 minutos (estabiliza membrana cardíaca imediatamente); "
            "(2) insulina regular 10 UI + glicose 50% 50mL IV (reduz K+ em 0,5-1,5 mEq/L em "
            "30-60min); (3) salbutamol 10-20mg inalatório (efeito aditivo à insulina); (4) "
            "bicarbonato de sódio se acidose metabólica concomitante; (5) resinas de troca "
            "(patiromer, ciclossilicato de zircônio) para redução definitiva; (6) diálise se "
            "refratária. Monitorar ECG continuamente. Fonte: KDIGO 2020."
        )
    },
    {
        "pergunta": "Como diagnosticar doença celíaca?",
        "resposta": (
            "O diagnóstico de doença celíaca segue etapas: (1) Sorologia: anti-transglutaminase "
            "tecidual IgA (anti-tTG IgA) com IgA total sérica (excluir deficiência de IgA); "
            "anti-gliadina deaminada IgA/IgG como alternativa; (2) Biópsia duodenal (endoscopia "
            "com pelo menos 4-6 fragmentos): achados de atrofia vilositária (Marsh 2-3), "
            "hiperplasia de criptas, aumento de linfócitos intraepiteliais; (3) Resposta à "
            "dieta sem glúten confirma diagnóstico. Tipagem HLA-DQ2/DQ8 tem alto valor preditivo "
            "negativo. O diagnóstico exige dieta com glúten em curso. Fonte: ACG Guidelines 2023."
        )
    },
    {
        "pergunta": "Quais são as principais causas de dor torácica não cardíaca?",
        "resposta": (
            "As principais causas de dor torácica não cardíaca incluem: Pulmonares: "
            "tromboembolismo pulmonar, pneumotórax, pneumonia, pleurite; Gastrointestinais: "
            "doença do refluxo gastroesofágico (causa mais comum de dor torácica não cardíaca), "
            "espasmo esofágico, úlcera péptica, pancreatite; Musculoesqueléticas: costocondrite "
            "(síndrome de Tietze), síndrome da costela escorregadia, fratura de costela; "
            "Psiquiátricas: transtorno do pânico, ansiedade, somatização; Outras: herpes-zoster, "
            "dissecção aórtica. A exclusão de causas cardíacas e TEP é prioritária. "
            "Fonte: UpToDate 2024."
        )
    },
    {
        "pergunta": "Como tratar a infecção urinária não complicada em mulher adulta?",
        "resposta": (
            "Na cistite não complicada em mulheres adultas saudáveis: (1) nitrofurantoína "
            "100mg de liberação modificada 2x/dia por 5 dias (primeira escolha); (2) "
            "fosfomicina 3g dose única; (3) trimetoprima-sulfametoxazol 160/800mg 2x/dia "
            "por 3 dias (se resistência local < 20%); (4) quinolonas devem ser evitadas como "
            "primeira linha (preservar para infecções mais graves). Urinocultura não é "
            "necessária em casos típicos. Piúria sem bacteriúria em mulheres jovens = "
            "investigar Chlamydia. Fonte: IDSA Guidelines 2011, revisão ESCMID 2022."
        )
    },
    {
        "pergunta": "Qual o manejo da crise hipertensiva com lesão de órgão-alvo?",
        "resposta": (
            "A emergência hipertensiva (PA elevada + lesão aguda de órgão-alvo) requer "
            "internação em UTI e redução gradual da PA: (1) reduzir PAM em 10-20% na "
            "primeira hora, 25% nas primeiras 2-6h, normalização em 24-48h (exceto AVC "
            "isquêmico: mais conservador); (2) Fármacos IV de escolha conforme manifestação: "
            "edema pulmonar agudo: nitroprussiato/nitroglicerina + furosemida; encefalopatia/PRES: "
            "nicardipina ou labetalol; síndrome coronariana aguda: nitroglicerina + beta-bloqueador; "
            "dissecção aórtica: labetalol ou esmolol + nitroprussiato. Fonte: ESC Hypertension "
            "Guidelines 2023."
        )
    },
    {
        "pergunta": "Como diagnosticar e tratar a gota?",
        "resposta": (
            "O diagnóstico de gota é feito por: (1) artrocentese com identificação de "
            "cristais de urato monossódico (birrefringência negativa à luz polarizada) - "
            "padrão-ouro; (2) critérios clínicos: artrite monoarticular aguda em 1º "
            "metatarsofalangiana (podagra), tofos, hiperuricemia. Tratamento da crise aguda: "
            "colchicina 1mg seguida de 0,5mg após 1h (preferencial), AINE (naproxeno, "
            "indometacina) ou corticosteroide. Profilaxia de crises + uricossurico: alopurinol "
            "100-600mg/dia (ajuste pela TFG) iniciado após resolução da crise, com meta de "
            "ácido úrico < 6mg/dL. Fonte: ACR Guidelines 2020."
        )
    },
    {
        "pergunta": "Quais são as indicações de internação na pneumonia adquirida na comunidade?",
        "resposta": (
            "A decisão de internar na PAC pode ser guiada pelo escore PSI/PORT (5 classes, "
            "classes IV-V = internação) ou CURB-65 (1 ponto cada: Confusão, Ureia > 7mmol/L, "
            "FR >= 30irpm, PA sistólica < 90 ou diastólica <= 60mmHg, Idade >= 65 anos): "
            "0-1 = ambulatorial; 2 = considerar internação breve; >= 3 = internação. "
            "Indicações adicionais: SatO2 < 90%, comorbidades descompensadas, falha do "
            "tratamento ambulatorial, impossibilidade de cuidado domiciliar. UTI se critérios "
            "maiores de ATS/IDSA. Fonte: IDSA/ATS Guidelines 2007."
        )
    },
    {
        "pergunta": "Como interpretar os padrões do líquido cefalorraquidiano?",
        "resposta": (
            "Padrões do LCR: Normal: aspecto cristalino, células <= 5/mm³ (linfócitos), "
            "proteína 15-45mg/dL, glicose 60-80% da glicemia. Meningite bacteriana: turvo, "
            "pleocitose intensa (> 1000 células, predominância PMN), proteína elevada (> 100mg/dL), "
            "glicose < 40mg/dL ou relação LCR/sangue < 0,4, Gram e cultura positivos. "
            "Meningite viral: límpido, pleocitose moderada (< 500 células, predominância "
            "linfócitos), proteína levemente elevada, glicose normal. Meningite tuberculosa: "
            "xantocrômico, pleocitose mista, proteína muito elevada, glicose baixa. "
            "Fonte: Harrison's Principles of Internal Medicine, 21ª ed."
        )
    },
    {
        "pergunta": "Quais são os critérios de risco para síncope cardiovascular?",
        "resposta": (
            "Fatores de alto risco na síncope que indicam investigação urgente ou internação: "
            "ECG anormal (bloqueio bifascicular, QTc prolongado, WPW, BRE novo, IAMCSST), "
            "síncope durante esforço ou deitado, ausência de pródromo, morte súbita familiar "
            "< 50 anos, sopro cardíaco novo, hipotensão ortostática severa. Escore ROSE: risco "
            "elevado se BNP > 300pg/mL, bradicardia < 50bpm, Hb < 9g/dL, Hgb fecal positiva, "
            "saturação < 94%, ou ECG anormal. A síncope vasovagal típica (gatilho, pródromo, "
            "jovem) tem prognóstico benigno. Fonte: ESC Guidelines on Syncope 2018."
        )
    },
    {
        "pergunta": "Como manejar a overdose de paracetamol?",
        "resposta": (
            "Na intoxicação por paracetamol: (1) Avaliação inicial: dose ingerida, tempo "
            "decorrido, dosagem sérica do paracetamol (nomograma de Rumack-Matthew para "
            "decisão de tratamento); (2) Descontaminação: carvão ativado 1g/kg VO se < 2-4h "
            "da ingestão; (3) Antídoto: N-acetilcisteína (NAC) - protocolo IV 21h: 150mg/kg "
            "em 200mL SF em 1h, 50mg/kg em 500mL em 4h, 100mg/kg em 1000mL em 16h; ou via "
            "oral: 140mg/kg ataque + 70mg/kg a cada 4h por 17 doses; (4) Monitorar: função "
            "hepática, INR, creatinina. NAC eficaz até 24h; transplante se insuficiência "
            "hepática fulminante. Fonte: UpToDate 2024."
        )
    },
    {
        "pergunta": "Quais são as causas de hipercalcemia e como tratá-la?",
        "resposta": (
            "Causas de hipercalcemia: PTH elevado - hiperparatireoidismo primário (adenoma, "
            "hiperplasia), terciário; PTH supresso - neoplasias (metástases ósseas, PTHrP), "
            "hipervitaminose D, sarcoidose e granulomatoses, imobilização, tiazídicos, "
            "síndrome leite-álcali. Tratamento (Ca > 12mg/dL ou sintomático): (1) soro "
            "fisiológico IV 200-300mL/h (hidratação vigorosa); (2) bisfosfonatos IV: "
            "zoledronato 4mg IV em 15 min (efeito em 24-72h, duração 4 semanas); (3) "
            "calcitonina 4-8 UI/kg IM/SC a cada 6-12h (efeito em 4-6h, taquifilaxia em "
            "48h); (4) corticosteroide em granulomatoses/linfoma; (5) diálise se refratária. "
            "Fonte: NEJM Review 2022."
        )
    },
    {
        "pergunta": "Como avaliar e manejar a dor abdominal aguda?",
        "resposta": (
            "A avaliação da dor abdominal aguda inclui: anamnese detalhada (localização, "
            "caráter, irradiação, fatores de melhora/piora, sintomas associados), exame "
            "físico completo (peritonismo, sinais especiais). Exames iniciais: hemograma, "
            "PCR, amilase/lipase, função renal e hepática, sumário de urina, beta-HCG "
            "(mulheres em idade fértil), radiografia de abdome, USG. TC abdominal se "
            "diagnóstico incerto ou alta suspeita de complicação. Sinais de abdome agudo "
            "cirúrgico (peritonite, perfusão comprometida) = cirurgia de urgência sem "
            "aguardar todos os exames. Fonte: ATLS 10ª edição."
        )
    },
    {
        "pergunta": "Quais são os princípios do suporte básico de vida (SBV) no adulto?",
        "resposta": (
            "O SBV no adulto segue a sequência: (1) Verificar segurança da cena; (2) "
            "Checar responsividade (estimulação verbal e tátil); (3) Acionar serviço de "
            "emergência e pedir DEA; (4) Checar pulso (carotídeo) por no máximo 10 segundos; "
            "(5) Se ausente: iniciar RCP - 30 compressões torácicas (5-6cm de profundidade, "
            "100-120/min, permitir reexpansão completa) : 2 ventilações de resgate; "
            "(6) Usar DEA assim que disponível (pausas mínimas < 10s); (7) Manter até SSVB "
            "ou chegada do SAMU. Alta qualidade da RCP é determinante para sobrevivência. "
            "Fonte: AHA Guidelines CPR 2020."
        )
    },
    {
        "pergunta": "Como diagnosticar hipotireoidismo?",
        "resposta": (
            "O diagnóstico de hipotireoidismo baseia-se em: (1) Clínica: fadiga, ganho de "
            "peso, intolerância ao frio, constipação, pele seca, bradicardia, mixedema, "
            "reflexos lentos; (2) Laboratorial: TSH elevado (principal triagem) + T4 livre "
            "reduzido = hipotireoidismo primário; TSH elevado + T4 livre normal = "
            "hipotireoidismo subclínico; TSH baixo/normal + T4 livre baixo = hipotireoidismo "
            "central (hipofisário/hipotalâmico). Investigar causa: anticorpos anti-TPO e "
            "anti-Tg (tireoidite de Hashimoto). Tratamento: levotiroxina com ajuste de dose "
            "pelo TSH. Fonte: ATA Guidelines 2014."
        )
    },
    {
        "pergunta": "Quais são os critérios para diagnóstico de artrite reumatoide?",
        "resposta": (
            "Os critérios ACR/EULAR 2010 para AR pontuam: (1) Acometimento articular: 1 "
            "grande articulação (0) a > 10 articulações incluindo pequenas (5); (2) "
            "Sorologia: FR e anti-CCP negativos (0), baixo positivo (2), alto positivo (3); "
            "(3) Reagentes de fase aguda: PCR e VHS normais (0), anormais (1); (4) Duração "
            "dos sintomas: < 6 semanas (0), >= 6 semanas (1). Pontuação >= 6 = AR definida. "
            "Investigar: FR, anti-CCP, PCR, VHS, hemograma, radiografia de mãos e pés. "
            "Início precoce do DMARD (metotrexato) melhora prognóstico articular. "
            "Fonte: ACR/EULAR 2010 Classification Criteria."
        )
    },
    {
        "pergunta": "Como manejar reação anafilática grave?",
        "resposta": (
            "A anafilaxia grave requer tratamento imediato: (1) ADRENALINA 0,3-0,5mg IM "
            "(face anterolateral da coxa) - repita a cada 5-15 minutos se necessário; é o "
            "tratamento de primeira linha e não deve ser retardado; (2) Decúbito dorsal com "
            "MMII elevados (exceto se vômitos/dispneia); (3) Oxigênio em alto fluxo; (4) "
            "Acesso venoso + SF 1-2L IV rápido se hipotensão; (5) Anti-histamínico: difenidramina "
            "25-50mg IV (adjuvante, não substitui adrenalina); (6) Corticosteroide: "
            "metilprednisolona 125mg IV (previne reação bifásica); (7) Observação mínima de "
            "4-6h. Prescrever adrenalina autoinjetável na alta. Fonte: WAO Anaphylaxis Guidelines 2020."
        )
    },
    {
        "pergunta": "Quais são as indicações de hemodiálise na insuficiência renal aguda?",
        "resposta": (
            "As indicações de terapia de substituição renal na IRA seguem o acrônimo AEIOU: "
            "Acidose metabólica grave (pH < 7,1) refratária; Eletrólitos: hipercalemia grave "
            "(K+ > 6,5 ou com alterações no ECG) refratária; Intoxicação por substâncias "
            "dialisáveis (metanol, etilenoglicol, lítio, salicilatos); Overload (sobrecarga "
            "volêmica): edema pulmonar refratário a diuréticos; Uremia sintomática: "
            "encefalopatia, pericardite, sangramento urêmico (ureia > 150-200mg/dL). "
            "A decisão deve ser individualizada e não baseada apenas em valores laboratoriais. "
            "Fonte: KDIGO AKI Guidelines 2012."
        )
    },
    {
        "pergunta": "Como avaliar e tratar a depressão maior?",
        "resposta": (
            "O diagnóstico de depressão maior (DSM-5) requer 5 ou mais sintomas por >= 2 "
            "semanas, incluindo obrigatoriamente humor deprimido e/ou anedonia: alteração "
            "do sono, apetite/peso, psicomotricidade, fadiga, sentimento de culpa/desvalia, "
            "dificuldade de concentração, ideação suicida. Ferramentas de triagem: PHQ-9 "
            "(score >= 10 = depressão moderada). Tratamento: leve-moderada: psicoterapia "
            "(TCC) + ISRS (sertralina, escitalopram, fluoxetina); grave: ISRS + TCC + "
            "considerar psiquiatria; risco suicida alto: hospitalização. Avaliar resposta "
            "em 4-8 semanas. Fonte: APA Practice Guidelines 2023."
        )
    },
    {
        "pergunta": "Quais são os princípios da prescrição segura de opioides para dor crônica?",
        "resposta": (
            "A prescrição segura de opioides para dor crônica inclui: (1) Estabelecer "
            "diagnóstico e falha de terapias não opioides; (2) Avaliar risco de abuso "
            "(ORT - Opioid Risk Tool); (3) Iniciar com dose mínima efetiva de opioide de "
            "ação curta; (4) Documentar: diagnóstico, metas funcionais, consentimento "
            "informado; (5) Contrato terapêutico: uso de uma única farmácia, não compartilhar; "
            "(6) Monitorar: reavaliação periódica da dor e função, urina toxicológica, "
            "vigilância de desvio; (7) Prescrever naloxona concomitantemente; (8) Não "
            "ultrapassar 90 MME/dia sem justificativa especializada. Fonte: CDC Opioid "
            "Prescribing Guideline 2022."
        )
    },
    {
        "pergunta": "Como identificar e manejar acidente vascular cerebral isquêmico agudo?",
        "resposta": (
            "O reconhecimento de AVC usa o acrônimo SAMU: Sorriso assimétrico, Abraço "
            "(fraqueza em um braço), fala com dificuldade, Urgência (ligar 192). Manejo "
            "no hospital: (1) TC de crânio sem contraste imediato (excluir hemorrágico); "
            "(2) glicemia capilar; (3) janela de 4,5h para trombólise IV com alteplase "
            "(0,9mg/kg, máx 90mg); (4) trombectomia mecânica até 24h em selecionados com "
            "oclusão de grande vaso; (5) AAS 300mg após excluir hemorragia; (6) PA: não "
            "tratar se < 220/120 em não candidatos a trombólise; (7) Unidade de AVC. "
            "Tempo = cérebro. Fonte: AHA/ASA Stroke Guidelines 2019."
        )
    },
    {
        "pergunta": "Quais são os critérios de alarme na dispepsia que indicam endoscopia?",
        "resposta": (
            "Os sinais de alarme na dispepsia que indicam endoscopia digestiva alta urgente "
            "incluem: idade > 55 anos com sintomas novos, disfagia ou odinofagia progressiva, "
            "vômitos persistentes ou recorrentes, perda de peso não intencional > 10%, "
            "sangramento gastrointestinal (hematêmese, melena, anemia ferropriva inexplicada), "
            "massa abdominal palpável ou linfadenopatia, histórico familiar de câncer gástrico "
            "ou esofágico em parente de primeiro grau, cirurgia gástrica prévia. Na ausência "
            "de alarmes, testar e tratar H. pylori ou prescrever IBP empírico por 4-8 semanas. "
            "Fonte: ACG Clinical Guideline 2017."
        )
    },
    {
        "pergunta": "Como prescrever anticoagulação na fibrilação atrial não valvar?",
        "resposta": (
            "A anticoagulação na FA não valvar é indicada pelo escore CHA2DS2-VASc: "
            "Insuficiência cardíaca, Hipertensão, Idade 65-74 anos, Diabetes mellitus, "
            "AVC/AIT/tromboembolismo prévio (2 pontos), Doença vascular, Sexo feminino "
            "(cada = 1 ponto, AVC prévio = 2). Homens >= 2 e mulheres >= 3 = anticoagular; "
            "homens com 1 ou mulheres com 2 = considerar. ACO de escolha: NOACs (dabigatrana, "
            "rivaroxabana, apixabana, edoxabana) são superiores à varfarina exceto em valvopatia "
            "mitral reumática ou válvula mecânica. Avaliar risco hemorrágico (HAS-BLED). "
            "Fonte: ESC Guidelines AF 2020."
        )
    },
    {
        "pergunta": "Como diagnosticar e tratar escabiose?",
        "resposta": (
            "O diagnóstico de escabiose (sarna) é clínico: prurido intenso de piora noturna, "
            "lesões papulovesiculares em áreas típicas (espaços interdigitais, punhos, "
            "região genital, axilas), presença de túneis (patognomônico). Confirmação: "
            "dermatoscopia (sinal do planador de asa delta) ou raspado de pele com "
            "visualização de ácaros/ovos. Tratamento de escolha: permetrina 5% creme "
            "aplicado da cervical para baixo, manter 8-12h, repetir em 7 dias; alternativa: "
            "ivermectina 200mcg/kg VO dose única, repetida em 2 semanas. Tratar todos os "
            "contactantes domiciliares simultaneamente. Lavar roupas e roupas de cama a 60°C. "
            "Fonte: AAD Guidelines 2021."
        )
    },
    {
        "pergunta": "Quais são os principais efeitos adversos dos antidepressivos ISRS?",
        "resposta": (
            "Os principais efeitos adversos dos ISRS incluem: Gastrointestinais (mais comuns "
            "no início): náusea, diarreia, xerostomia, constipação - geralmente transitórios; "
            "Sexuais: disfunção erétil, anorgasmia, diminuição da libido (afetam até 40-70% "
            "dos pacientes); SNC: cefaleia, insônia, agitação, sonolência, tremor; Síndrome "
            "serotoninérgica: raro mas grave - hipertermia, rigidez, mioclonias, confusão "
            "(risco aumentado com IMAO, tramadol, linezolida); Síndrome de descontinuação: "
            "fluoxetina tem menor risco pela meia-vida longa; Hiponatremia (idosos); "
            "Aumento do risco de sangramento (interação com AAS/AINE). "
            "Fonte: Goodman & Gilman 13ª ed."
        )
    },
    {
        "pergunta": "Como diagnosticar e tratar síndrome do túnel do carpo?",
        "resposta": (
            "A síndrome do túnel do carpo (STC) apresenta-se com parestesias e dor nos "
            "3 primeiros dedos e metade radial do 4º, piora noturna e com atividades com "
            "punho em flexão. Testes: Phalen (flexão do punho por 60s reproduz sintomas) e "
            "Tinel (percussão sobre o túnel do carpo). Confirmação: eletroneuromiografia "
            "(atraso na velocidade de condução do nervo mediano). Tratamento: leve-moderado: "
            "órtese noturna de punho em posição neutra + fisioterapia + infiltração com "
            "corticosteroide; grave ou refratário: descompressão cirúrgica do túnel do carpo "
            "(secção do ligamento transverso do carpo). Fonte: AAOS Guidelines 2016."
        )
    },
    {
        "pergunta": "Quais são as causas e o manejo de hipotensão ortostática?",
        "resposta": (
            "Hipotensão ortostática (HO) = queda de PA sistólica >= 20mmHg ou diastólica "
            ">= 10mmHg em 3 minutos da posição supina para ortostase. Causas: "
            "Neurogênica: Parkinson, neuropatia autonômica diabética, amiloidose, "
            "atrofia de múltiplos sistemas; Não neurogênica: hipovolemia, insuficiência "
            "adrenal, medicamentos (anti-hipertensivos, diuréticos, antidepressivos, "
            "alfa-bloqueadores, levodopa). Manejo: tratar causa base, suspender/reduzir "
            "medicamentos causais, expandir volemia (sal + água), meias de compressão, "
            "elevar cabeceira 30°; farmacológico: fludrocortisona 0,1mg/dia ou midodrina "
            "2,5-10mg 3x/dia. Fonte: AAN Practice Parameter 2017."
        )
    },
    {
        "pergunta": "Como rastrear e prevenir câncer colorretal?",
        "resposta": (
            "O rastreamento de câncer colorretal em adultos de risco médio (sem fatores de "
            "risco especiais) deve ser iniciado aos 45 anos. Opções: (1) Colonoscopia a "
            "cada 10 anos (preferencial - diagnóstica e terapêutica); (2) Pesquisa de sangue "
            "oculto nas fezes de alta sensibilidade (FIT ou gFOBT) anualmente; (3) "
            "Sigmoidoscopia flexível a cada 5 anos; (4) FIT-DNA (Cologuard) a cada 1-3 anos; "
            "(5) TC colonoscopia a cada 5 anos. Risco aumentado (história familiar, DII, "
            "pólipos prévios): iniciar mais cedo e com menor intervalo. Prevenção: dieta "
            "rica em fibras, AAS (em selecionados), cessação do tabagismo. "
            "Fonte: ACG/USPSTF Guidelines 2021."
        )
    },
    {
        "pergunta": "Como tratar a crise de enxaqueca aguda?",
        "resposta": (
            "O tratamento da crise de enxaqueca segue abordagem escalonada: (1) Analgésicos "
            "simples: AAS 1000mg, ibuprofeno 400-600mg, naproxeno 500-1000mg, paracetamol "
            "1000mg (para crises leves-moderadas); (2) Triptanos: sumatriptano 50-100mg VO "
            "(ou 6mg SC, 20mg intranasal), rizatriptano 10mg, eletriptano 40mg (para crises "
            "moderadas-graves ou falha de analgésicos); (3) Antieméticos: metoclopramida "
            "10mg IV/IM (também tem efeito na enxaqueca); (4) Evitar opioides (risco de "
            "cronificação). Profilaxia indicada se > 4 crises/mês: propranolol, amitriptilina, "
            "topiramato, valproato, CGRP antagonistas. Fonte: AHS/AAN Guidelines 2019."
        )
    },
    {
        "pergunta": "Quais são os critérios diagnósticos para síndrome metabólica?",
        "resposta": (
            "A síndrome metabólica (critérios IDF/AHA/NHLBI 2009) requer 3 dos 5 componentes: "
            "(1) Circunferência abdominal aumentada: >= 90cm homens e >= 80cm mulheres "
            "(critérios para população latino-americana/asiática); >= 102cm e >= 88cm para "
            "caucasianos; (2) Triglicerídeos >= 150mg/dL ou em tratamento; (3) HDL-c < 40mg/dL "
            "em homens ou < 50mg/dL em mulheres, ou em tratamento; (4) PA >= 130/85mmHg ou "
            "em tratamento anti-hipertensivo; (5) Glicemia de jejum >= 100mg/dL ou em "
            "tratamento para DM. Associada a risco cardiovascular e DM2 aumentados. "
            "Fonte: Joint Interim Statement 2009."
        )
    },
    {
        "pergunta": "Como manejar dor lombar aguda inespecífica?",
        "resposta": (
            "A dor lombar aguda inespecífica (< 6 semanas) é autolimitada em 90% dos casos. "
            "Manejo: (1) Tranquilizar o paciente sobre prognóstico favorável; (2) Manter "
            "atividade física dentro do tolerável (evitar repouso absoluto); (3) Analgesia: "
            "paracetamol ou AINEs em dose plena por tempo limitado (primeira linha), miorrelaxantes "
            "podem ser associados por curto prazo; (4) Exercício físico ativo (fisioterapia "
            "se não melhora em 2-4 semanas); (5) Investigar red flags: déficit neurológico "
            "progressivo, síndrome da cauda equina, suspeita de neoplasia/infecção/fratura. "
            "Imagem (RM) não indicada rotineiramente nas primeiras 6 semanas. "
            "Fonte: ACP Clinical Guideline 2017."
        )
    },
    {
        "pergunta": "Quais são as indicações de imunização na gestação?",
        "resposta": (
            "Vacinas indicadas na gestação (calendário SBIm/PNI 2024): (1) dTpa (difteria, "
            "tétano, coqueluche acelular): a cada gestação, entre 20-36 semanas, para "
            "proteção do neonato (estratégia casulo); (2) Influenza inativada: a qualquer "
            "momento da gestação, dose anual; (3) Hepatite B: se não vacinada, esquema "
            "completo 0-1-6 meses; (4) COVID-19 (inativada ou mRNA): indicada em qualquer "
            "trimestre. Contraindicadas (vírus vivos atenuados): febre amarela (exceto em "
            "áreas de risco elevado), tríplice viral (SCR), varicela, BCG. Vacinas inativadas "
            "são seguras na gestação. Fonte: PNI/SBIm 2024."
        )
    },
    {
        "pergunta": "Como diagnosticar insuficiência adrenal?",
        "resposta": (
            "O diagnóstico de insuficiência adrenal (IA): (1) Suspeitar em: fraqueza, fadiga, "
            "perda de peso, hipotensão, hiponatremia, hipercalemia, hipoglicemia, hiperpigmentação "
            "cutânea (IA primária); (2) Screening: cortisol sérico matinal (7-9h): < 5 mcg/dL "
            "= IA provável, > 18 mcg/dL = exclui; zona cinzenta 5-18 mcg/dL = teste de "
            "estimulação; (3) Teste de estimulação com ACTH (250 mcg IV): cortisol pós-estímulo "
            "> 18-20 mcg/dL = normal; (4) ACTH plasmático: elevado na primária, baixo na "
            "secundária; (5) Crise adrenal: tratamento imediato com hidrocortisona 100mg IV. "
            "Fonte: Endocrine Society Clinical Practice Guideline 2016."
        )
    },
    {
        "pergunta": "Quais são os principais diagnósticos diferenciais do exantema febril?",
        "resposta": (
            "Os principais diagnósticos diferenciais do exantema febril incluem: "
            "Virais: sarampo (exantema maculopapular cefalocaudal, manchas de Koplik), "
            "rubéola (adenopatia retroauricular), roséola infantil/HHV-6 (exantema após "
            "defervescência), varicela (vesículas pruriginosas em surtos), dengue "
            "(exantema petequial), Zika (exantema fino), chikungunya; Bacterianas: "
            "escarlatina (exantema em lixa, sinal de Pastia), meningococcemia (petéquias/ "
            "púrpura - emergência), febre maculosa (petéquias em extremidades); "
            "Medicamentosas: farmacodermia; Outras: doença de Kawasaki (< 5 anos), "
            "síndrome de Stevens-Johnson. Fonte: Nelson Textbook of Pediatrics, 21ª ed."
        )
    },
    {
        "pergunta": "Como prevenir e tratar úlceras por pressão?",
        "resposta": (
            "Prevenção de lesões por pressão: (1) Avaliação de risco (escala de Braden <= "
            "18 = risco); (2) Reposicionamento a cada 2h em acamados; (3) Superfícies "
            "especiais (colchões de espuma viscoelástica ou pneumáticos para alto risco); "
            "(4) Hidratação da pele e proteção de proeminências ósseas; (5) Nutrição "
            "adequada (proteína 1,2-1,5g/kg/dia). Tratamento conforme estadiamento: "
            "Estágio I (eritema): alívio de pressão, filme transparente; Estágio II "
            "(perda parcial): cobertura hidrocoloide ou hidrogel; Estágio III-IV "
            "(perda total): desbridamento, curativo avançado (alginato, espuma), "
            "considerar cirurgia. Fonte: NPUAP/EPUAP/PPPIA Guidelines 2019."
        )
    },
    {
        "pergunta": "Como diagnosticar e tratar conjuntivite bacteriana?",
        "resposta": (
            "A conjuntivite bacteriana apresenta: secreção purulenta ou mucopurulenta, "
            "hiperemia conjuntival, sensação de corpo estranho/ardor (sem dor intensa ou "
            "perda visual). Agentes comuns: S. aureus, S. pneumoniae, H. influenzae, "
            "M. catarrhalis. Neonatos: N. gonorrhoeae (hiperaguda, 2-5 dias) e C. trachomatis "
            "(5-14 dias). Tratamento: adultos: colírio de ciprofloxacino 0,3% ou tobramicina "
            "0,3% 4x/dia por 5-7 dias; neonatal por gonococo: ceftriaxona 25-50mg/kg IV "
            "dose única; por clamídia: eritromicina VO por 14 dias. Resolução espontânea "
            "em 7-14 dias sem tratamento nos casos leves. "
            "Fonte: AAO Preferred Practice Pattern 2018."
        )
    },
]


def baixar_pubmedqa() -> list[dict]:
    """
    Baixa e processa o dataset PubMedQA da HuggingFace.

    Retorna lista de dicionários no formato de instrução.
    """
    logger.info("Baixando dataset PubMedQA (qiaojin/PubMedQA, pqa_labeled)...")

    try:
        dataset = load_dataset(DATASET_PUBMEDQA, CONFIG_PUBMEDQA, trust_remote_code=True)
        exemplos_processados = []

        for split_nome, split_dados in dataset.items():
            logger.info(f"Processando split '{split_nome}' com {len(split_dados)} exemplos...")

            for exemplo in split_dados:
                pergunta = exemplo.get("question", "").strip()
                resposta_longa = exemplo.get("long_answer", "").strip()
                contexto = exemplo.get("context", {})

                if not pergunta or not resposta_longa:
                    continue

                # Extrai contexto adicional se disponível
                contexto_texto = ""
                if isinstance(contexto, dict) and "contexts" in contexto:
                    contexto_texto = " ".join(contexto["contexts"][:2])

                entrada = pergunta
                if contexto_texto:
                    entrada = f"{pergunta}\n\nContexto: {contexto_texto[:300]}..."

                exemplos_processados.append({
                    "instruction": INSTRUCAO_SISTEMA,
                    "input": anonymize_text(entrada),
                    "output": anonymize_text(resposta_longa[:800])
                })

        logger.info(f"PubMedQA: {len(exemplos_processados)} exemplos carregados")
        return exemplos_processados

    except Exception as erro:
        logger.warning(f"Falha ao baixar PubMedQA: {erro}")
        logger.warning("Continuando sem o dataset PubMedQA...")
        return []


def anonymize_text(texto: str) -> str:
    """
    Remove informações pessoais identificáveis do texto médico.

    Aplica regex para remover:
    - Nomes próprios de pacientes (padrão: 'Patient John Smith')
    - Datas no formato DD/MM/YYYY, MM/DD/YYYY e variações
    - Números de registro médico (MRN, ID)
    - Números de telefone
    - Endereços de e-mail

    Parâmetros:
        texto: String com texto potencialmente contendo dados sensíveis

    Retorna:
        String com dados sensíveis substituídos por marcadores
    """
    if not isinstance(texto, str):
        return str(texto)

    # Remove referências a pacientes com nome
    texto = re.sub(
        r'\b(?:patient|paciente|pt\.?)\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*\b',
        '[PACIENTE]',
        texto,
        flags=re.IGNORECASE
    )

    # Remove datas no formato DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
    texto = re.sub(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        '[DATA]',
        texto
    )

    # Remove anos isolados que parecem datas de nascimento (ex: "born 1985")
    texto = re.sub(
        r'\b(?:born|nascido|DOB|dob)\s+(?:in\s+)?\d{4}\b',
        '[DATA_NASCIMENTO]',
        texto,
        flags=re.IGNORECASE
    )

    # Remove números de registro médico (MRN, ID, #)
    texto = re.sub(
        r'\b(?:MRN|Medical Record Number|ID|registro)\s*[:#]?\s*\d{5,10}\b',
        '[ID_REGISTRO]',
        texto,
        flags=re.IGNORECASE
    )

    # Remove números de telefone
    texto = re.sub(
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{3,5}[-.\s]?\d{4}\b',
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


def gerar_exemplos_sinteticos() -> list[dict]:
    """
    Gera exemplos sintéticos de perguntas médicas em português.

    Usa uma base de pares pergunta-resposta predefinidos, expandidos com
    variações para aumentar a diversidade do dataset.

    Retorna:
        Lista de dicionários no formato de instrução para fine-tuning
    """
    logger.info(f"Gerando {NUM_EXEMPLOS_SINTETICOS} exemplos sintéticos em português...")

    random.seed(SEMENTE_ALEATORIA)
    exemplos_sinteticos = []

    # Usa todos os pares disponíveis
    for par in PARES_SINTETICOS:
        exemplos_sinteticos.append({
            "instruction": INSTRUCAO_SISTEMA,
            "input": par["pergunta"],
            "output": par["resposta"]
        })

    # Se precisar de mais exemplos, usa variações dos existentes
    variantes_instrucao = [
        "Você é um assistente médico especializado. Responda de forma clara e baseada em evidências:",
        "Como assistente de apoio clínico, responda a seguinte pergunta médica:",
        "Você é um assistente médico. Forneça informações baseadas em diretrizes clínicas:",
    ]

    while len(exemplos_sinteticos) < NUM_EXEMPLOS_SINTETICOS:
        par_base = random.choice(PARES_SINTETICOS)
        instrucao_variante = random.choice(variantes_instrucao)
        exemplos_sinteticos.append({
            "instruction": instrucao_variante,
            "input": par_base["pergunta"],
            "output": par_base["resposta"]
        })

    # Limita ao número desejado e embaralha
    exemplos_sinteticos = exemplos_sinteticos[:NUM_EXEMPLOS_SINTETICOS]
    random.shuffle(exemplos_sinteticos)

    logger.info(f"Gerados {len(exemplos_sinteticos)} exemplos sintéticos")
    return exemplos_sinteticos


def limpar_texto(texto: str, comprimento_maximo: int = 1000) -> str:
    """
    Limpa e padroniza o texto removendo caracteres problemáticos.

    Parâmetros:
        texto: Texto a ser limpo
        comprimento_maximo: Tamanho máximo em caracteres

    Retorna:
        Texto limpo e truncado se necessário
    """
    if not isinstance(texto, str):
        texto = str(texto)

    # Remove caracteres de controle exceto quebras de linha normais
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)

    # Normaliza espaços em branco
    texto = re.sub(r'\s+', ' ', texto).strip()

    # Trunca se necessário
    if len(texto) > comprimento_maximo:
        texto = texto[:comprimento_maximo] + "..."

    return texto


def validar_exemplo(exemplo: dict) -> bool:
    """
    Valida se um exemplo tem os campos obrigatórios e conteúdo mínimo.

    Parâmetros:
        exemplo: Dicionário com campos instruction, input e output

    Retorna:
        True se o exemplo é válido, False caso contrário
    """
    campos_obrigatorios = ["instruction", "input", "output"]

    for campo in campos_obrigatorios:
        if campo not in exemplo:
            return False
        if not isinstance(exemplo[campo], str):
            return False
        if len(exemplo[campo].strip()) < 10:
            return False

    return True


def imprimir_estatisticas(dataset: list[dict]) -> None:
    """
    Imprime estatísticas do dataset processado para monitoramento.

    Parâmetros:
        dataset: Lista de exemplos processados
    """
    if not dataset:
        logger.warning("Dataset vazio - sem estatísticas para exibir")
        return

    tamanhos_output = [len(ex["output"]) for ex in dataset]
    tamanhos_input = [len(ex["input"]) for ex in dataset]

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DO DATASET PROCESSADO")
    print("=" * 60)
    print(f"Total de exemplos:           {len(dataset)}")
    print(f"Tamanho médio das respostas: {sum(tamanhos_output) / len(tamanhos_output):.0f} caracteres")
    print(f"Tamanho máximo das respostas:{max(tamanhos_output)} caracteres")
    print(f"Tamanho mínimo das respostas:{min(tamanhos_output)} caracteres")
    print(f"Tamanho médio das perguntas: {sum(tamanhos_input) / len(tamanhos_input):.0f} caracteres")
    print("=" * 60 + "\n")


def salvar_dataset(dataset: list[dict], caminho: Path) -> None:
    """
    Salva o dataset processado em formato JSON.

    Parâmetros:
        dataset: Lista de exemplos a salvar
        caminho: Caminho do arquivo de saída
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(dataset, arquivo, ensure_ascii=False, indent=2)

    logger.info(f"Dataset salvo em: {caminho} ({len(dataset)} exemplos)")


def preparar_dataset() -> None:
    """
    Função principal que orquestra o pipeline de preparação do dataset.

    Executa em sequência:
    1. Download do PubMedQA
    2. Geração de dados sintéticos
    3. Validação e limpeza
    4. Salvamento do dataset final
    """
    logger.info("Iniciando pipeline de preparação do dataset médico...")

    todos_exemplos = []

    # Etapa 1: Baixar PubMedQA
    exemplos_pubmed = baixar_pubmedqa()
    todos_exemplos.extend(exemplos_pubmed)

    # Etapa 2: Gerar exemplos sintéticos em português
    exemplos_sinteticos = gerar_exemplos_sinteticos()
    todos_exemplos.extend(exemplos_sinteticos)

    # Etapa 3: Limpar e validar cada exemplo
    logger.info("Limpando e validando exemplos...")
    exemplos_validos = []

    for exemplo in todos_exemplos:
        # Limpa os textos
        exemplo["instruction"] = limpar_texto(exemplo["instruction"], comprimento_maximo=200)
        exemplo["input"] = limpar_texto(exemplo["input"], comprimento_maximo=800)
        exemplo["output"] = limpar_texto(exemplo["output"], comprimento_maximo=1000)

        # Valida o exemplo
        if validar_exemplo(exemplo):
            exemplos_validos.append(exemplo)

    logger.info(f"Exemplos válidos após validação: {len(exemplos_validos)}/{len(todos_exemplos)}")

    # Etapa 4: Embaralhar para distribuição aleatória
    random.seed(SEMENTE_ALEATORIA)
    random.shuffle(exemplos_validos)

    # Etapa 5: Salvar datasets
    salvar_dataset(exemplos_validos, CAMINHO_SAIDA)

    # Salva também os dados sintéticos separadamente
    salvar_dataset(exemplos_sinteticos, CAMINHO_SINTETICOS)

    # Etapa 6: Imprimir estatísticas finais
    imprimir_estatisticas(exemplos_validos)

    logger.info("Pipeline de preparação concluído com sucesso!")


if __name__ == "__main__":
    preparar_dataset()
