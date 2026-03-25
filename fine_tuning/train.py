"""
train.py
--------
Fine-tuning do LLaMA 3 8B com QLoRA (4-bit quantization).
Usa SFTTrainer da biblioteca TRL para treinamento supervisionado.

Configuração QLoRA:
- Quantização: 4-bit (NF4)
- LoRA rank: 16
- LoRA alpha: 32
- Dropout: 0.05
- Target modules: q_proj, v_proj

Uso:
    python fine_tuning/train.py
    python fine_tuning/train.py --config fine_tuning/config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================================
# CONSTANTES
# ============================================================
CAMINHO_CONFIG_PADRAO = "fine_tuning/config.yaml"
CAMINHO_DATASET_PADRAO = "data/processed/medical_dataset.json"
CAMINHO_SAIDA_PADRAO = "models/medical-llama3-qlora"
CAMINHO_GRAFICO_LOSS = "logs/training_loss.png"
SEMENTE_ALEATORIA = 42

# Configura o logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def carregar_configuracao(caminho_config: str) -> dict:
    """
    Carrega as configurações de treinamento do arquivo YAML.

    Parâmetros:
        caminho_config: Caminho para o arquivo config.yaml

    Retorna:
        Dicionário com todas as configurações de treinamento
    """
    logger.info(f"Carregando configurações de: {caminho_config}")

    with open(caminho_config, "r", encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)

    logger.info("Configurações carregadas com sucesso")
    return config


def verificar_gpu() -> bool:
    """
    Verifica disponibilidade de GPU e exibe informações do hardware.

    Retorna:
        True se GPU disponível, False para CPU
    """
    if torch.cuda.is_available():
        nome_gpu = torch.cuda.get_device_name(0)
        memoria_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detectada: {nome_gpu} ({memoria_total:.1f} GB VRAM)")
        return True
    else:
        logger.warning("GPU não detectada. Treinamento será feito na CPU (muito lento).")
        return False


def configurar_quantizacao(config: dict) -> BitsAndBytesConfig:
    """
    Configura a quantização 4-bit (QLoRA) usando BitsAndBytes.

    Parâmetros:
        config: Dicionário com configurações de quantização

    Retorna:
        Objeto BitsAndBytesConfig configurado
    """
    config_quant = config.get("quantization", {})

    dtype_mapa = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tipo_computacao = dtype_mapa.get(
        config_quant.get("bnb_4bit_compute_dtype", "float16"),
        torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config_quant.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=config_quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config_quant.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("Configuração de quantização 4-bit (QLoRA) configurada")
    return bnb_config


def carregar_modelo_e_tokenizador(config: dict, bnb_config: BitsAndBytesConfig):
    """
    Carrega o modelo base e o tokenizador com quantização 4-bit.

    Tenta carregar o modelo principal e usa fallback se falhar.

    Parâmetros:
        config: Dicionário com configurações do modelo
        bnb_config: Configuração de quantização BitsAndBytes

    Retorna:
        Tupla (model, tokenizer) carregados e preparados
    """
    config_modelo = config.get("model", {})
    nome_modelo = config_modelo.get("name", "mistralai/Mistral-7B-v0.1")
    nome_fallback = config_modelo.get("fallback", "mistralai/Mistral-7B-v0.1")
    token_hf = os.environ.get("HUGGINGFACE_TOKEN")

    # Tenta carregar o modelo principal, usa fallback em caso de erro
    for nome in [nome_modelo, nome_fallback]:
        try:
            logger.info(f"Carregando modelo: {nome}")

            tokenizador = AutoTokenizer.from_pretrained(
                nome,
                token=token_hf,
                trust_remote_code=True
            )

            # Configura o token de padding
            if tokenizador.pad_token is None:
                tokenizador.pad_token = tokenizador.eos_token

            modelo = AutoModelForCausalLM.from_pretrained(
                nome,
                quantization_config=bnb_config,
                device_map="auto",
                token=token_hf,
                trust_remote_code=True,
            )

            logger.info(f"Modelo '{nome}' carregado com sucesso")
            return modelo, tokenizador

        except Exception as erro:
            logger.warning(f"Falha ao carregar '{nome}': {erro}")
            if nome == nome_fallback:
                raise RuntimeError(
                    f"Não foi possível carregar nenhum modelo. "
                    f"Verifique HUGGINGFACE_TOKEN e conexão com internet."
                ) from erro

    return None, None


def configurar_lora(config: dict) -> LoraConfig:
    """
    Configura os adaptadores LoRA para fine-tuning eficiente.

    Parâmetros:
        config: Dicionário com configurações LoRA

    Retorna:
        Objeto LoraConfig configurado
    """
    config_lora = config.get("lora", {})

    lora_config = LoraConfig(
        r=config_lora.get("r", 16),
        lora_alpha=config_lora.get("lora_alpha", 32),
        lora_dropout=config_lora.get("lora_dropout", 0.05),
        target_modules=config_lora.get("target_modules", ["q_proj", "v_proj"]),
        bias=config_lora.get("bias", "none"),
        task_type=config_lora.get("task_type", "CAUSAL_LM"),
    )

    logger.info(
        f"LoRA configurado: rank={lora_config.r}, "
        f"alpha={lora_config.lora_alpha}, "
        f"módulos={lora_config.target_modules}"
    )
    return lora_config


def carregar_dataset_treinamento(caminho_dataset: str, config: dict) -> tuple:
    """
    Carrega e divide o dataset em conjuntos de treino e validação.

    Parâmetros:
        caminho_dataset: Caminho para o arquivo JSON do dataset
        config: Dicionário com configurações do dataset

    Retorna:
        Tupla (dataset_treino, dataset_validacao)
    """
    logger.info(f"Carregando dataset de: {caminho_dataset}")

    config_dataset = config.get("dataset", {})
    proporcao_validacao = config_dataset.get("validation_split", 0.1)
    semente = config_dataset.get("seed", SEMENTE_ALEATORIA)
    max_amostras = config_dataset.get("max_train_samples")

    with open(caminho_dataset, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    # Limita amostras se especificado
    if max_amostras and max_amostras < len(dados):
        dados = dados[:max_amostras]
        logger.info(f"Limitado a {max_amostras} exemplos de treino")

    # Cria dataset HuggingFace
    dataset = Dataset.from_list(dados)

    # Divide em treino e validação
    divisao = dataset.train_test_split(
        test_size=proporcao_validacao,
        seed=semente
    )

    logger.info(
        f"Dataset: {len(divisao['train'])} exemplos de treino, "
        f"{len(divisao['test'])} exemplos de validação"
    )

    return divisao["train"], divisao["test"]


def formatar_exemplo_alpaca(exemplo: dict) -> str:
    """
    Formata um exemplo no template Alpaca para o SFTTrainer.

    Parâmetros:
        exemplo: Dicionário com instruction, input e output

    Retorna:
        String formatada no padrão Alpaca
    """
    instrucao = exemplo.get("instruction", "")
    entrada = exemplo.get("input", "")
    saida = exemplo.get("output", "")

    if entrada:
        texto = (
            f"### Instrução:\n{instrucao}\n\n"
            f"### Pergunta:\n{entrada}\n\n"
            f"### Resposta:\n{saida}"
        )
    else:
        texto = (
            f"### Instrução:\n{instrucao}\n\n"
            f"### Resposta:\n{saida}"
        )

    return texto


def salvar_grafico_loss(historico_treino: list, caminho_saida: str) -> None:
    """
    Gera e salva o gráfico de curva de loss do treinamento.

    Parâmetros:
        historico_treino: Lista de logs com métricas de treino
        caminho_saida: Caminho para salvar o gráfico PNG
    """
    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)

    # Extrai valores de loss dos logs
    passos = [log["step"] for log in historico_treino if "loss" in log]
    losses = [log["loss"] for log in historico_treino if "loss" in log]

    if not passos:
        logger.warning("Sem dados de loss para plotar")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(passos, losses, "b-o", linewidth=2, markersize=4, label="Loss de Treinamento")
    plt.xlabel("Passos de Treinamento", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Curva de Loss — Fine-tuning do Modelo Médico (QLoRA)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Gráfico de loss salvo em: {caminho_saida}")


def executar_treinamento(config: dict) -> None:
    """
    Função principal que executa o pipeline completo de fine-tuning.

    Etapas:
    1. Verificação de hardware (GPU)
    2. Configuração de quantização QLoRA
    3. Carregamento do modelo e tokenizador
    4. Configuração dos adaptadores LoRA
    5. Carregamento do dataset
    6. Configuração e execução do SFTTrainer
    7. Salvamento do modelo e gráfico de loss

    Parâmetros:
        config: Dicionário completo de configurações
    """
    # Etapa 1: Verificar hardware
    tem_gpu = verificar_gpu()

    # Etapa 2: Configurar quantização
    bnb_config = configurar_quantizacao(config)

    # Etapa 3: Carregar modelo e tokenizador
    modelo, tokenizador = carregar_modelo_e_tokenizador(config, bnb_config)

    # Prepara o modelo para treinamento com k-bit
    modelo = prepare_model_for_kbit_training(modelo)

    # Etapa 4: Configurar LoRA
    lora_config = configurar_lora(config)
    modelo = get_peft_model(modelo, lora_config)

    # Exibe parâmetros treináveis
    modelo.print_trainable_parameters()

    # Etapa 5: Carregar dataset
    config_paths = config.get("paths", {})
    caminho_dataset = config_paths.get("dataset", CAMINHO_DATASET_PADRAO)
    caminho_saida = config_paths.get("output_model", CAMINHO_SAIDA_PADRAO)
    caminho_logs = config_paths.get("output_logs", "logs/")
    caminho_grafico = config_paths.get("loss_plot", CAMINHO_GRAFICO_LOSS)

    dataset_treino_raw, dataset_validacao_raw = carregar_dataset_treinamento(caminho_dataset, config)

    # Etapa 6: Pré-formata o dataset aplicando o template Alpaca
    # No TRL >= 0.15, formatting_func foi removido do SFTTrainer;
    # o dataset deve ter um campo "text" já formatado.
    def mapear_para_texto(exemplo):
        return {"text": formatar_exemplo_alpaca(exemplo)}

    dataset_treino = dataset_treino_raw.map(mapear_para_texto, remove_columns=dataset_treino_raw.column_names)
    dataset_validacao = dataset_validacao_raw.map(mapear_para_texto, remove_columns=dataset_validacao_raw.column_names)

    config_treino = config.get("training", {})
    config_modelo = config.get("model", {})
    comprimento_max = config_modelo.get("max_seq_length", 2048)
    warmup_steps = max(1, int(
        (len(dataset_treino) // config_treino.get("per_device_train_batch_size", 4))
        * config_treino.get("num_train_epochs", 3)
        * config_treino.get("warmup_ratio", 0.02)
    ))

    from trl import SFTConfig

    argumentos_treino = SFTConfig(
        output_dir=caminho_saida,
        num_train_epochs=config_treino.get("num_train_epochs", 3),
        per_device_train_batch_size=config_treino.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config_treino.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config_treino.get("gradient_accumulation_steps", 4),
        learning_rate=config_treino.get("learning_rate", 2e-4),
        lr_scheduler_type=config_treino.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        save_steps=config_treino.get("save_steps", 50),
        eval_steps=config_treino.get("eval_steps", 50),
        eval_strategy="steps",
        save_total_limit=config_treino.get("save_total_limit", 3),
        logging_steps=config_treino.get("logging_steps", 10),
        gradient_checkpointing=config_treino.get("gradient_checkpointing", True),
        fp16=False,
        bf16=tem_gpu,
        optim=config_treino.get("optim", "paged_adamw_32bit"),
        report_to="none",
        load_best_model_at_end=True,
        seed=SEMENTE_ALEATORIA,
        max_length=comprimento_max,
        dataset_text_field="text",
        packing=False,
    )

    # Etapa 7: Inicializar e executar SFTTrainer (API TRL 0.29.x)
    logger.info("Inicializando SFTTrainer...")

    treinador = SFTTrainer(
        model=modelo,
        args=argumentos_treino,
        train_dataset=dataset_treino,
        eval_dataset=dataset_validacao,
        processing_class=tokenizador,
    )

    logger.info("Iniciando treinamento...")
    resultado = treinador.train()

    logger.info(f"Treinamento concluído! Loss final: {resultado.training_loss:.4f}")

    # Etapa 8: Salvar modelo e tokenizador
    Path(caminho_saida).mkdir(parents=True, exist_ok=True)
    treinador.save_model(caminho_saida)
    tokenizador.save_pretrained(caminho_saida)
    logger.info(f"Modelo salvo em: {caminho_saida}")

    # Etapa 9: Salvar gráfico de loss
    historico = treinador.state.log_history
    salvar_grafico_loss(historico, caminho_grafico)

    logger.info("Pipeline de fine-tuning concluído com sucesso!")


def main() -> None:
    """
    Ponto de entrada principal do script de treinamento.
    Analisa argumentos da linha de comando e inicia o treinamento.
    """
    analisador = argparse.ArgumentParser(
        description="Fine-tuning de LLM médico com QLoRA"
    )
    analisador.add_argument(
        "--config",
        type=str,
        default=CAMINHO_CONFIG_PADRAO,
        help="Caminho para o arquivo de configuração YAML"
    )
    argumentos = analisador.parse_args()

    # Verifica se o arquivo de config existe
    if not Path(argumentos.config).exists():
        logger.error(f"Arquivo de configuração não encontrado: {argumentos.config}")
        sys.exit(1)

    # Verifica se o dataset existe
    config = carregar_configuracao(argumentos.config)
    caminho_dataset = config.get("paths", {}).get("dataset", CAMINHO_DATASET_PADRAO)

    if not Path(caminho_dataset).exists():
        logger.error(
            f"Dataset não encontrado em '{caminho_dataset}'. "
            f"Execute primeiro: python fine_tuning/prepare_dataset.py"
        )
        sys.exit(1)

    # Inicia o treinamento
    executar_treinamento(config)


if __name__ == "__main__":
    main()
