"""
test_modules.py
---------------
Testes rápidos de todos os módulos da aplicação.
Não requer GPU nem modelo treinado — valida a lógica dos componentes.

Uso:
    python tests/test_modules.py
"""

import sys
import json
import tempfile
import traceback
from pathlib import Path

# Adiciona a raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

VERDE = "\033[92m"
VERMELHO = "\033[91m"
AMARELO = "\033[93m"
RESET = "\033[0m"
NEGRITO = "\033[1m"

# Configura encoding UTF-8 no Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

resultados = {"ok": 0, "falhou": 0, "avisos": 0}


def ok(mensagem: str) -> None:
    print(f"  {VERDE}[OK]{RESET} {mensagem}")
    resultados["ok"] += 1


def falhou(mensagem: str, erro: str = "") -> None:
    print(f"  {VERMELHO}[FALHOU]{RESET} {mensagem}")
    if erro:
        print(f"    {VERMELHO}-> {erro}{RESET}")
    resultados["falhou"] += 1


def aviso(mensagem: str) -> None:
    print(f"  {AMARELO}[AVISO]{RESET} {mensagem}")
    resultados["avisos"] += 1


def secao(titulo: str) -> None:
    print(f"\n{NEGRITO}{'-'*55}{RESET}")
    print(f"{NEGRITO} {titulo}{RESET}")
    print(f"{NEGRITO}{'-'*55}{RESET}")


# ============================================================
# TESTE 1: IMPORTAÇÕES BÁSICAS
# ============================================================
def testar_importacoes_basicas():
    secao("1. Importações básicas (sem GPU)")

    modulos = [
        ("json", "json"),
        ("re", "re"),
        ("pathlib", "pathlib"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
    ]

    for modulo, pacote in modulos:
        try:
            __import__(modulo)
            ok(f"{modulo}")
        except ImportError:
            falhou(f"{modulo}", f"Instale: pip install {pacote}")


# ============================================================
# TESTE 2: IMPORTAÇÕES ML (aviso se não instalado)
# ============================================================
def testar_importacoes_ml():
    secao("2. Importações ML / LangChain")

    modulos_ml = [
        ("torch", "torch>=2.1.0"),
        ("transformers", "transformers>=4.40.0"),
        ("peft", "peft>=0.10.0"),
        ("datasets", "datasets>=2.19.0"),
        ("trl", "trl>=0.8.0"),
        ("langchain", "langchain>=0.2.0"),
        ("langgraph", "langgraph>=0.1.0"),
        ("faiss", "faiss-cpu>=1.8.0"),
        ("sentence_transformers", "sentence-transformers>=2.7.0"),
        ("gradio", "gradio>=4.31.0"),
        ("rouge_score", "rouge-score>=0.1.2"),
        ("huggingface_hub", "huggingface_hub"),
    ]

    for modulo, pacote in modulos_ml:
        try:
            __import__(modulo)
            ok(f"{modulo}")
        except ImportError:
            aviso(f"{modulo} não instalado — instale: pip install {pacote}")


# ============================================================
# TESTE 3: MÓDULO SAFETY
# ============================================================
def testar_safety():
    secao("3. Módulo de Segurança (safety.py)")

    try:
        from assistant.safety import ValidadorSeguranca, RESPOSTA_FORA_ESCOPO

        validador = ValidadorSeguranca()

        # Teste 1: pergunta médica válida deve passar
        resultado = validador.validar_resposta(
            pergunta="Qual o tratamento para hipertensão?",
            resposta="O tratamento inclui mudanças no estilo de vida.",
            chain_utilizada="teste"
        )
        assert resultado.resposta_final != "", "Resposta não deve ser vazia"
        assert RESPOSTA_FORA_ESCOPO not in resultado.resposta_final
        ok("Pergunta médica válida aprovada")

        # Teste 2: pergunta fora do escopo deve ser bloqueada
        resultado2 = validador.validar_resposta(
            pergunta="Me ajude a cozinhar um bolo de chocolate",
            resposta="Aqui está a receita...",
            chain_utilizada="teste"
        )
        assert resultado2.fora_do_escopo is True
        ok("Pergunta fora do escopo bloqueada corretamente")

        # Teste 3: palavras de emergência detectadas
        resultado3 = validador.validar_resposta(
            pergunta="Paciente com infarto agudo do miocárdio",
            resposta="Conduta imediata recomendada.",
            chain_utilizada="teste"
        )
        assert resultado3.e_emergencia is True
        ok("Palavra-chave de emergência detectada")

        # Teste 4: linguagem diagnóstica definitiva deve ser suavizada
        resultado4 = validador.validar_resposta(
            pergunta="O que o paciente tem?",
            resposta="Você tem diabetes mellitus tipo 2.",
            chain_utilizada="teste"
        )
        assert "você tem" not in resultado4.resposta_final.lower()
        ok("Linguagem diagnóstica definitiva suavizada")

        # Teste 5: disclaimer sempre presente
        assert "não substitui" in resultado.resposta_final.lower() or \
               "aviso" in resultado.resposta_final.lower() or \
               "aviso importante" in resultado.resposta_final.lower()
        ok("Disclaimer obrigatório incluído na resposta")

    except ImportError as e:
        falhou("Importação do safety.py falhou", str(e))
    except AssertionError as e:
        falhou("Asserção do safety.py falhou", str(e))
    except Exception as e:
        falhou("Erro inesperado no safety.py", str(e))


# ============================================================
# TESTE 4: MÓDULO LOGGER
# ============================================================
def testar_logger():
    secao("4. Módulo de Auditoria (logger.py)")

    try:
        from assistant.logger import AuditoriaLogger

        with tempfile.TemporaryDirectory() as pasta_temp:
            caminho_log = str(Path(pasta_temp) / "test_interactions.jsonl")
            logger = AuditoriaLogger(caminho_log=caminho_log)

            # Teste 1: registra interação corretamente
            id_interacao = logger.registrar_interacao(
                pergunta_usuario="Qual o tratamento para HAS?",
                resposta_assistente="Mudanças no estilo de vida...",
                chain_utilizada="clinical_qa",
                fontes_citadas=["Diretriz SBH 2020"],
                flag_alerta=False
            )
            assert id_interacao is not None and len(id_interacao) > 0
            ok("Interação registrada com ID único")

            # Teste 2: arquivo JSONL criado
            assert Path(caminho_log).exists()
            ok("Arquivo JSONL criado em disco")

            # Teste 3: conteúdo é JSON válido
            with open(caminho_log, "r", encoding="utf-8") as f:
                linha = f.readline()
                dados = json.loads(linha)
            assert "timestamp" in dados
            assert "chain_utilizada" in dados
            assert "pergunta_anonimizada" in dados
            ok("Conteúdo do log é JSON válido com campos obrigatórios")

            # Teste 4: anonimização funciona
            logger.registrar_interacao(
                pergunta_usuario="Paciente João Silva, CPF 123.456.789-00, nascido em 01/01/1980",
                resposta_assistente="Resposta de teste",
                chain_utilizada="teste"
            )
            interacoes = logger.recuperar_interacoes_recentes(5)
            ultima = interacoes[-1]["pergunta_anonimizada"]
            assert "João Silva" not in ultima, "Nome não foi anonimizado"
            assert "123.456.789-00" not in ultima, "CPF não foi anonimizado"
            ok("Anonimização de nome e CPF funcionando")

            # Teste 5: recupera interações recentes
            interacoes = logger.recuperar_interacoes_recentes(10)
            assert len(interacoes) == 2
            ok(f"Recuperação de interações recentes: {len(interacoes)} encontradas")

            # Teste 6: estatísticas calculadas
            stats = logger.obter_estatisticas()
            assert "total_interacoes" in stats
            assert stats["total_interacoes"] == 2
            ok(f"Estatísticas calculadas: {stats['total_interacoes']} interações")

    except ImportError as e:
        falhou("Importação do logger.py falhou", str(e))
    except AssertionError as e:
        falhou("Asserção do logger.py falhou", str(e))
    except Exception as e:
        falhou("Erro inesperado no logger.py", str(e))


# ============================================================
# TESTE 5: PREPARAÇÃO DO DATASET (sintéticos apenas)
# ============================================================
def testar_prepare_dataset():
    secao("5. Preparação do Dataset (exemplos sintéticos)")

    try:
        # Importa apenas as funções que não dependem de 'datasets' (HuggingFace)
        import importlib, types
        # Mock do módulo 'datasets' caso não esteja instalado
        if "datasets" not in sys.modules:
            mock = types.ModuleType("datasets")
            mock.load_dataset = lambda *a, **kw: {}
            sys.modules["datasets"] = mock

        from fine_tuning.prepare_dataset import (
            anonymize_text,
            gerar_exemplos_sinteticos,
            validar_exemplo,
            limpar_texto,
            PARES_SINTETICOS,
        )

        # Teste 1: geração de sintéticos
        exemplos = gerar_exemplos_sinteticos()
        assert len(exemplos) == 50, f"Esperado 50, obtido {len(exemplos)}"
        ok(f"Gerados {len(exemplos)} exemplos sintéticos")

        # Teste 2: formato correto de cada exemplo
        for ex in exemplos[:5]:
            assert "instruction" in ex
            assert "input" in ex
            assert "output" in ex
        ok("Formato instruction/input/output correto")

        # Teste 3: anonimização
        texto_com_dados = "Patient John Smith, born 01/01/1980, MRN: 123456"
        texto_limpo = anonymize_text(texto_com_dados)
        assert "John Smith" not in texto_limpo
        assert "01/01/1980" not in texto_limpo
        assert "123456" not in texto_limpo
        ok("Anonimização de dados sensíveis funcionando")

        # Teste 4: validação de exemplos (mínimo 10 chars por campo)
        exemplo_valido = {"instruction": "Você é um assistente médico.", "input": "Qual o tratamento?", "output": "O tratamento inclui..."}
        assert validar_exemplo(exemplo_valido) is True
        exemplo_invalido = {"instruction": "a", "input": "", "output": "ok"}
        assert validar_exemplo(exemplo_invalido) is False
        ok("Validação de exemplos (válido/inválido) funcionando")

        # Teste 5: limpeza de texto
        texto_sujo = "  texto   com   espaços   \n\n extras  "
        texto_limpo2 = limpar_texto(texto_sujo)
        assert "  " not in texto_limpo2
        ok("Limpeza e normalização de texto funcionando")

        # Teste 6: todos os pares sintéticos têm conteúdo
        assert len(PARES_SINTETICOS) >= 40
        ok(f"{len(PARES_SINTETICOS)} pares médicos carregados na base sintética")

    except ImportError as e:
        falhou("Importação do prepare_dataset.py falhou", str(e))
    except AssertionError as e:
        falhou("Asserção do prepare_dataset.py falhou", str(e))
    except Exception as e:
        falhou("Erro inesperado no prepare_dataset.py", str(e))


# ============================================================
# TESTE 6: CONFIGURAÇÕES YAML
# ============================================================
def testar_config_yaml():
    secao("6. Arquivo de Configuração (config.yaml)")

    try:
        import yaml

        caminho = Path("fine_tuning/config.yaml")
        assert caminho.exists(), "config.yaml não encontrado"

        with open(caminho, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Valida seções obrigatórias
        secoes_obrigatorias = ["model", "quantization", "lora", "training", "paths", "dataset"]
        for secao_nome in secoes_obrigatorias:
            assert secao_nome in config, f"Seção '{secao_nome}' ausente"
        ok("Todas as seções obrigatórias presentes")

        # Valida hiperparâmetros críticos
        assert config["lora"]["r"] == 16
        assert config["lora"]["lora_alpha"] == 32
        assert config["training"]["learning_rate"] == 0.0002
        assert config["training"]["per_device_train_batch_size"] == 4
        ok("Hiperparâmetros QLoRA corretos (rank=16, alpha=32, lr=2e-4)")

        # Valida caminhos
        assert "output_model" in config["paths"]
        assert "dataset" in config["paths"]
        ok("Caminhos de saída configurados")

    except FileNotFoundError as e:
        falhou("config.yaml não encontrado", str(e))
    except AssertionError as e:
        falhou("Validação do config.yaml falhou", str(e))
    except Exception as e:
        falhou("Erro inesperado no config.yaml", str(e))


# ============================================================
# TESTE 7: VARIÁVEIS DE AMBIENTE
# ============================================================
def testar_env():
    secao("7. Variáveis de Ambiente (.env)")

    try:
        from dotenv import load_dotenv
        import os

        env_path = Path(".env")
        env_example_path = Path(".env.example")

        assert env_example_path.exists(), ".env.example não encontrado"
        ok(".env.example encontrado")

        if env_path.exists():
            load_dotenv(env_path)
            ok(".env encontrado e carregado")

            token = os.environ.get("HUGGINGFACE_TOKEN", "")
            if token and token != "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
                ok("HUGGINGFACE_TOKEN configurado")
            else:
                aviso("HUGGINGFACE_TOKEN não configurado ou usando valor de exemplo")
        else:
            aviso(".env não encontrado — copie de .env.example e configure")

    except Exception as e:
        falhou("Erro ao verificar .env", str(e))


# ============================================================
# TESTE 8: ESTRUTURA DE PASTAS
# ============================================================
def testar_estrutura_pastas():
    secao("8. Estrutura de Pastas e Arquivos")

    arquivos_obrigatorios = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "Makefile",
        "docker-compose.yml",
        "fine_tuning/config.yaml",
        "fine_tuning/prepare_dataset.py",
        "fine_tuning/train.py",
        "fine_tuning/evaluate.py",
        "fine_tuning/upload_model.py",
        "assistant/chains.py",
        "assistant/graph.py",
        "assistant/retriever.py",
        "assistant/safety.py",
        "assistant/logger.py",
        "interface/app.py",
        "docker/Dockerfile.training",
        "docker/Dockerfile.app",
        "docker/nginx.conf",
        "models/.gitkeep",
        "logs/.gitkeep",
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "data/synthetic/.gitkeep",
    ]

    for arquivo in arquivos_obrigatorios:
        if Path(arquivo).exists():
            ok(arquivo)
        else:
            falhou(arquivo, "Arquivo não encontrado")


# ============================================================
# TESTE 9: GPU (informativo)
# ============================================================
def testar_gpu():
    secao("9. Disponibilidade de GPU (informativo)")

    try:
        import torch

        if torch.cuda.is_available():
            nome = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            versao_cuda = torch.version.cuda
            ok(f"GPU detectada: {nome}")
            ok(f"VRAM disponível: {vram:.1f} GB")
            ok(f"Versão CUDA: {versao_cuda}")

            if vram >= 12:
                ok("VRAM suficiente para fine-tuning QLoRA")
            elif vram >= 6:
                aviso(f"VRAM ({vram:.0f}GB) suficiente para inferência, mas pode ser limitada para treino")
            else:
                aviso(f"VRAM ({vram:.0f}GB) pode ser insuficiente para QLoRA — use batch_size menor")
        else:
            aviso("GPU não detectada — fine-tuning não disponível")
            aviso("A aplicação usará modelo de demonstração (microsoft/phi-2)")

    except ImportError:
        aviso("PyTorch não instalado — não foi possível verificar GPU")


# ============================================================
# RELATÓRIO FINAL
# ============================================================
def imprimir_relatorio():
    total = resultados["ok"] + resultados["falhou"]
    print(f"\n{'='*55}")
    print(f"{NEGRITO} RESULTADO FINAL{RESET}")
    print(f"{'='*55}")
    print(f"  {VERDE}Passou:  {resultados['ok']}/{total}{RESET}")
    if resultados["falhou"] > 0:
        print(f"  {VERMELHO}Falhou:  {resultados['falhou']}/{total}{RESET}")
    if resultados["avisos"] > 0:
        print(f"  {AMARELO}Avisos:  {resultados['avisos']}{RESET}")
    print(f"{'='*55}\n")

    if resultados["falhou"] == 0:
        print(f"{VERDE}{NEGRITO}  Todos os testes passaram!{RESET}\n")
        return 0
    else:
        print(f"{VERMELHO}{NEGRITO}  {resultados['falhou']} teste(s) falharam. Verifique os erros acima.{RESET}\n")
        return 1


if __name__ == "__main__":
    print(f"\n{NEGRITO}{'='*55}{RESET}")
    print(f"{NEGRITO}  NEURIX — Testes do Assistente Médico Virtual{RESET}")
    print(f"{NEGRITO}{'='*55}{RESET}")

    testar_importacoes_basicas()
    testar_importacoes_ml()
    testar_safety()
    testar_logger()
    testar_prepare_dataset()
    testar_config_yaml()
    testar_env()
    testar_estrutura_pastas()
    testar_gpu()

    sys.exit(imprimir_relatorio())
