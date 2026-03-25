"""
upload_model.py
---------------
Faz o upload do modelo fine-tunado para o HuggingFace Hub.

Uso:
    python fine_tuning/upload_model.py
    python fine_tuning/upload_model.py --model models/medical-llama3-qlora --repo seu-usuario/medical-llama3
    python fine_tuning/upload_model.py --private   # repositório privado
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# CONSTANTES
# ============================================================
CAMINHO_MODELO_PADRAO = "models/medical-llama3-qlora"
REPO_PADRAO = "theladycode/NEURIX"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def autenticar_huggingface(token: str) -> None:
    """
    Autentica no HuggingFace Hub com o token fornecido.

    Parâmetros:
        token: Token de acesso HuggingFace (write permission)
    """
    logger.info("Autenticando no HuggingFace Hub...")
    login(token=token)
    logger.info("Autenticação bem-sucedida")


def obter_nome_usuario(token: str) -> str:
    """
    Obtém o nome de usuário da conta HuggingFace autenticada.

    Parâmetros:
        token: Token de acesso

    Retorna:
        Nome de usuário no HuggingFace
    """
    api = HfApi()
    info = api.whoami(token=token)
    return info["name"]


def fazer_upload_modelo(
    caminho_modelo: str,
    nome_repo: str,
    token: str,
    privado: bool = False
) -> str:
    """
    Faz o upload do modelo e tokenizador para o HuggingFace Hub.

    Parâmetros:
        caminho_modelo: Caminho local do modelo treinado
        nome_repo: Nome do repositório no formato 'usuario/nome-modelo'
        token: Token HuggingFace com permissão de escrita
        privado: True para repositório privado

    Retorna:
        URL do modelo no HuggingFace Hub
    """
    logger.info(f"Carregando modelo de: {caminho_modelo}")

    tokenizador = AutoTokenizer.from_pretrained(caminho_modelo, trust_remote_code=True)

    logger.info(f"Fazendo upload para: https://huggingface.co/{nome_repo}")
    logger.info(f"Repositório: {'privado' if privado else 'público'}")

    # Upload do tokenizador
    tokenizador.push_to_hub(
        nome_repo,
        token=token,
        private=privado,
    )
    logger.info("Tokenizador enviado")

    # Upload do modelo completo (adaptadores LoRA + config)
    # Usa push_to_hub direto da pasta salva (mais rápido que recarregar o modelo base)
    api = HfApi()
    api.upload_folder(
        folder_path=caminho_modelo,
        repo_id=nome_repo,
        repo_type="model",
        token=token,
        commit_message="Upload do modelo médico fine-tunado com QLoRA — FIAP PosTech",
    )

    url_modelo = f"https://huggingface.co/{nome_repo}"
    logger.info(f"Upload concluído! Modelo disponível em: {url_modelo}")
    return url_modelo


def criar_model_card(nome_repo: str, caminho_modelo: str) -> str:
    """
    Cria o README (Model Card) do modelo no HuggingFace.

    Parâmetros:
        nome_repo: Nome do repositório
        caminho_modelo: Caminho local do modelo

    Retorna:
        Conteúdo do Model Card em Markdown
    """
    return f"""---
language:
- pt
- en
license: llama3
base_model: meta-llama/Meta-Llama-3-8B
tags:
- medical
- healthcare
- fine-tuned
- qlora
- peft
- portuguese
pipeline_tag: text-generation
---

# Medical LLaMA 3 — Assistente Médico Virtual (QLoRA)

Modelo fine-tunado sobre o **LLaMA 3 8B** com técnica **QLoRA (4-bit)** para
apoio à decisão clínica em português.

## Sobre o Modelo

Este modelo foi desenvolvido como projeto acadêmico de pós-graduação (FIAP PosTech)
e é um **sistema de apoio à decisão clínica**. NÃO substitui avaliação médica profissional.

## Como Usar

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("{nome_repo}")
model = AutoModelForCausalLM.from_pretrained("{nome_repo}")

prompt = \"\"\"### Instrução:
Você é um assistente médico. Responda a pergunta clínica:

### Pergunta:
Qual o tratamento de primeira linha para hipertensão arterial estágio 1?

### Resposta:\"\"\"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Treinamento

- **Modelo base:** meta-llama/Meta-Llama-3-8B
- **Técnica:** QLoRA (4-bit NF4 + LoRA rank 16)
- **Dataset:** PubMedQA + exemplos sintéticos em português
- **Framework:** TRL SFTTrainer + HuggingFace PEFT

## Aviso

Este modelo é um **protótipo acadêmico**. As respostas geradas devem ser
validadas por profissional de saúde habilitado antes de qualquer aplicação clínica.
"""


def main() -> None:
    """
    Ponto de entrada: analisa argumentos e executa o upload.
    """
    analisador = argparse.ArgumentParser(
        description="Upload do modelo médico fine-tunado para o HuggingFace Hub"
    )
    analisador.add_argument(
        "--model",
        type=str,
        default=CAMINHO_MODELO_PADRAO,
        help="Caminho local do modelo treinado"
    )
    analisador.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Nome do repositório (ex: meu-usuario/medical-llama3). "
             "Se omitido, usa o nome de usuário HuggingFace automaticamente."
    )
    analisador.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Criar repositório privado (padrão: público)"
    )
    argumentos = analisador.parse_args()

    # Verifica se o modelo existe
    if not Path(argumentos.model).exists():
        logger.error(
            f"Modelo não encontrado em '{argumentos.model}'. "
            f"Execute primeiro: python fine_tuning/train.py"
        )
        sys.exit(1)

    # Obtém o token do ambiente
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.error(
            "HUGGINGFACE_TOKEN não encontrado. "
            "Defina no arquivo .env ou exporte a variável de ambiente."
        )
        sys.exit(1)

    # Autentica no Hub
    autenticar_huggingface(token)

    # Define o nome do repositório
    if argumentos.repo:
        nome_repo = argumentos.repo
    else:
        nome_repo = REPO_PADRAO
        logger.info(f"Repositório padrão: {nome_repo}")

    # Salva o Model Card localmente antes do upload
    model_card_path = Path(argumentos.model) / "README.md"
    model_card_path.write_text(
        criar_model_card(nome_repo, argumentos.model),
        encoding="utf-8"
    )
    logger.info(f"Model Card criado para: {nome_repo}")
    logger.info("Model Card criado")

    # Faz o upload
    url = fazer_upload_modelo(
        caminho_modelo=argumentos.model,
        nome_repo=nome_repo,
        token=token,
        privado=argumentos.private,
    )

    print(f"\n{'='*60}")
    print(f"  UPLOAD CONCLUIDO!")
    print(f"  Modelo disponivel em: {url}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
