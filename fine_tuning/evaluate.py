"""
evaluate.py
-----------
Avaliação do modelo médico fine-tunado com métricas quantitativas e qualitativas.

Métricas utilizadas:
- ROUGE-1, ROUGE-2, ROUGE-L: para qualidade das respostas geradas
- Perplexidade: medida de fluência do modelo
- Avaliação qualitativa: comparação visual entre resposta gerada e referência

Uso:
    python fine_tuning/evaluate.py
    python fine_tuning/evaluate.py --model models/medical-llama3-qlora --num_samples 20
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONSTANTES
# ============================================================
CAMINHO_MODELO_PADRAO = "models/medical-llama3-qlora"
CAMINHO_DATASET_PADRAO = "data/processed/medical_dataset.json"
CAMINHO_RELATORIO = "logs/evaluation_report.json"
CAMINHO_TABELA = "logs/evaluation_results.csv"
NUM_AMOSTRAS_PADRAO = 10
MAX_NOVOS_TOKENS = 512
TEMPERATURA_GERACAO = 0.7

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def carregar_modelo_avaliacao(caminho_modelo: str):
    """
    Carrega o modelo fine-tunado para avaliação com quantização 4-bit.

    Parâmetros:
        caminho_modelo: Caminho para o diretório do modelo salvo

    Retorna:
        Tupla (model, tokenizer) prontos para inferência
    """
    logger.info(f"Carregando modelo para avaliação: {caminho_modelo}")
    token_hf = os.environ.get("HUGGINGFACE_TOKEN")

    # Configuração de quantização para inferência eficiente
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Lê o adapter_config.json para descobrir o modelo base
        adapter_config_path = Path(caminho_modelo) / "adapter_config.json"
        if adapter_config_path.exists():
            import json as _json
            with open(adapter_config_path, "r") as f:
                adapter_cfg = _json.load(f)
            nome_base = adapter_cfg.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.1")
            logger.info(f"Modelo base detectado: {nome_base}")
        else:
            nome_base = caminho_modelo

        tokenizador = AutoTokenizer.from_pretrained(
            caminho_modelo,
            token=token_hf,
            trust_remote_code=True
        )

        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token

        # Carrega o modelo base com quantização
        modelo_base = AutoModelForCausalLM.from_pretrained(
            nome_base,
            quantization_config=bnb_config,
            device_map="auto",
            token=token_hf,
            trust_remote_code=True,
        )

        # Aplica os adaptadores LoRA salvos
        if adapter_config_path.exists():
            modelo = PeftModel.from_pretrained(modelo_base, caminho_modelo)
            logger.info("Adaptadores LoRA carregados com sucesso")
        else:
            modelo = modelo_base

        modelo.eval()
        logger.info("Modelo carregado com sucesso para avaliação")
        return modelo, tokenizador

    except Exception as erro:
        logger.error(f"Erro ao carregar modelo: {erro}")
        raise


def gerar_resposta(
    modelo,
    tokenizador,
    instrucao: str,
    pergunta: str,
    max_novos_tokens: int = MAX_NOVOS_TOKENS
) -> str:
    """
    Gera uma resposta do modelo para uma dada instrução e pergunta.

    Parâmetros:
        modelo: Modelo LLM carregado
        tokenizador: Tokenizador correspondente
        instrucao: Instrução do sistema médico
        pergunta: Pergunta clínica a ser respondida
        max_novos_tokens: Número máximo de tokens na resposta

    Retorna:
        String com a resposta gerada pelo modelo
    """
    # Formata o prompt no template Alpaca
    prompt = (
        f"### Instrução:\n{instrucao}\n\n"
        f"### Pergunta:\n{pergunta}\n\n"
        f"### Resposta:\n"
    )

    entradas = tokenizador(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(modelo.device)

    with torch.no_grad():
        ids_gerados = modelo.generate(
            **entradas,
            max_new_tokens=max_novos_tokens,
            temperature=TEMPERATURA_GERACAO,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizador.eos_token_id,
        )

    # Decodifica apenas os novos tokens gerados
    novos_ids = ids_gerados[0][entradas["input_ids"].shape[1]:]
    resposta = tokenizador.decode(novos_ids, skip_special_tokens=True)

    return resposta.strip()


def calcular_metricas_rouge(
    respostas_geradas: list[str],
    respostas_referencia: list[str]
) -> dict:
    """
    Calcula as métricas ROUGE para avaliar qualidade das respostas geradas.

    Parâmetros:
        respostas_geradas: Lista de respostas geradas pelo modelo
        respostas_referencia: Lista de respostas de referência (ground truth)

    Retorna:
        Dicionário com médias de ROUGE-1, ROUGE-2 e ROUGE-L
    """
    logger.info("Calculando métricas ROUGE...")
    avaliador = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores_acumulados = {
        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
        "rouge2": {"precision": [], "recall": [], "fmeasure": []},
        "rougeL": {"precision": [], "recall": [], "fmeasure": []},
    }

    for gerada, referencia in zip(respostas_geradas, respostas_referencia):
        scores = avaliador.score(referencia, gerada)
        for metrica_nome, valores in scores.items():
            scores_acumulados[metrica_nome]["precision"].append(valores.precision)
            scores_acumulados[metrica_nome]["recall"].append(valores.recall)
            scores_acumulados[metrica_nome]["fmeasure"].append(valores.fmeasure)

    # Calcula médias
    resultados = {}
    for metrica_nome, valores in scores_acumulados.items():
        resultados[metrica_nome] = {
            "precision": float(np.mean(valores["precision"])),
            "recall": float(np.mean(valores["recall"])),
            "fmeasure": float(np.mean(valores["fmeasure"])),
        }

    return resultados


def avaliar_qualitativamente(
    amostras: list[dict],
    respostas_geradas: list[str]
) -> list[dict]:
    """
    Cria uma comparação qualitativa entre respostas geradas e referências.

    Parâmetros:
        amostras: Lista de exemplos do dataset com respostas de referência
        respostas_geradas: Lista de respostas geradas pelo modelo

    Retorna:
        Lista de dicionários com comparações par a par
    """
    comparacoes = []

    for i, (amostra, resposta_gerada) in enumerate(zip(amostras, respostas_geradas)):
        comparacao = {
            "indice": i + 1,
            "pergunta": amostra.get("input", ""),
            "resposta_referencia": amostra.get("output", ""),
            "resposta_gerada": resposta_gerada,
            "comprimento_referencia": len(amostra.get("output", "")),
            "comprimento_gerado": len(resposta_gerada),
        }
        comparacoes.append(comparacao)

    return comparacoes


def imprimir_relatorio(metricas_rouge: dict, comparacoes: list[dict]) -> None:
    """
    Imprime o relatório de avaliação de forma estruturada no console.

    Parâmetros:
        metricas_rouge: Dicionário com métricas ROUGE calculadas
        comparacoes: Lista de comparações qualitativas
    """
    print("\n" + "=" * 70)
    print("RELATÓRIO DE AVALIAÇÃO — ASSISTENTE MÉDICO (QLoRA)")
    print("=" * 70)

    print("\n📊 MÉTRICAS ROUGE (escala 0-1, quanto maior melhor):")
    print("-" * 50)
    for metrica, valores in metricas_rouge.items():
        print(f"  {metrica.upper()}:")
        print(f"    Precisão:    {valores['precision']:.4f}")
        print(f"    Recall:      {valores['recall']:.4f}")
        print(f"    F1 (média):  {valores['fmeasure']:.4f}")

    print("\n📝 EXEMPLOS QUALITATIVOS:")
    print("-" * 50)

    for comp in comparacoes[:3]:  # Mostra apenas 3 exemplos no console
        print(f"\nExemplo {comp['indice']}:")
        print(f"  Pergunta:    {comp['pergunta'][:100]}...")
        print(f"  Referência:  {comp['resposta_referencia'][:150]}...")
        print(f"  Gerado:      {comp['resposta_gerada'][:150]}...")

    print("\n" + "=" * 70)


def salvar_relatorio(
    metricas_rouge: dict,
    comparacoes: list[dict],
    num_amostras: int
) -> None:
    """
    Salva o relatório de avaliação em formatos JSON e CSV.

    Parâmetros:
        metricas_rouge: Dicionário com métricas ROUGE
        comparacoes: Lista de comparações qualitativas
        num_amostras: Número total de amostras avaliadas
    """
    Path(CAMINHO_RELATORIO).parent.mkdir(parents=True, exist_ok=True)

    # Salva relatório JSON completo
    relatorio = {
        "num_amostras_avaliadas": num_amostras,
        "metricas_rouge": metricas_rouge,
        "comparacoes_qualitativas": comparacoes,
    }

    with open(CAMINHO_RELATORIO, "w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)

    logger.info(f"Relatório JSON salvo em: {CAMINHO_RELATORIO}")

    # Salva tabela CSV das comparações
    df = pd.DataFrame(comparacoes)
    df.to_csv(CAMINHO_TABELA, index=False, encoding="utf-8")

    logger.info(f"Tabela CSV salva em: {CAMINHO_TABELA}")


def executar_avaliacao(caminho_modelo: str, caminho_dataset: str, num_amostras: int) -> None:
    """
    Pipeline principal de avaliação do modelo fine-tunado.

    Parâmetros:
        caminho_modelo: Caminho para o modelo salvo
        caminho_dataset: Caminho para o dataset de avaliação
        num_amostras: Número de exemplos a avaliar
    """
    logger.info("Iniciando pipeline de avaliação do modelo médico...")

    # Carrega o modelo
    modelo, tokenizador = carregar_modelo_avaliacao(caminho_modelo)

    # Carrega amostras do dataset para avaliação
    logger.info(f"Carregando {num_amostras} amostras do dataset...")
    with open(caminho_dataset, "r", encoding="utf-8") as arquivo:
        todos_dados = json.load(arquivo)

    # Pega as últimas amostras (não vistas no treino tipicamente)
    amostras_avaliacao = todos_dados[-num_amostras:]

    # Gera respostas para cada amostra
    logger.info("Gerando respostas do modelo...")
    respostas_geradas = []
    respostas_referencia = []

    for i, amostra in enumerate(amostras_avaliacao):
        logger.info(f"Processando amostra {i + 1}/{num_amostras}...")

        resposta_gerada = gerar_resposta(
            modelo,
            tokenizador,
            instrucao=amostra.get("instruction", ""),
            pergunta=amostra.get("input", "")
        )

        respostas_geradas.append(resposta_gerada)
        respostas_referencia.append(amostra.get("output", ""))

    # Calcula métricas ROUGE
    metricas_rouge = calcular_metricas_rouge(respostas_geradas, respostas_referencia)

    # Avaliação qualitativa
    comparacoes = avaliar_qualitativamente(amostras_avaliacao, respostas_geradas)

    # Imprime e salva relatório
    imprimir_relatorio(metricas_rouge, comparacoes)
    salvar_relatorio(metricas_rouge, comparacoes, num_amostras)

    logger.info("Avaliação concluída com sucesso!")


def main() -> None:
    """
    Ponto de entrada principal do script de avaliação.
    """
    analisador = argparse.ArgumentParser(
        description="Avaliação do modelo médico fine-tunado"
    )
    analisador.add_argument(
        "--model",
        type=str,
        default=CAMINHO_MODELO_PADRAO,
        help="Caminho para o modelo fine-tunado"
    )
    analisador.add_argument(
        "--dataset",
        type=str,
        default=CAMINHO_DATASET_PADRAO,
        help="Caminho para o dataset de avaliação"
    )
    analisador.add_argument(
        "--num_samples",
        type=int,
        default=NUM_AMOSTRAS_PADRAO,
        help="Número de amostras para avaliação"
    )
    argumentos = analisador.parse_args()

    # Verifica se o modelo existe
    if not Path(argumentos.model).exists():
        logger.error(
            f"Modelo não encontrado em '{argumentos.model}'. "
            f"Execute primeiro: python fine_tuning/train.py"
        )
        sys.exit(1)

    # Verifica se o dataset existe
    if not Path(argumentos.dataset).exists():
        logger.error(
            f"Dataset não encontrado em '{argumentos.dataset}'. "
            f"Execute primeiro: python fine_tuning/prepare_dataset.py"
        )
        sys.exit(1)

    executar_avaliacao(argumentos.model, argumentos.dataset, argumentos.num_samples)


if __name__ == "__main__":
    main()
