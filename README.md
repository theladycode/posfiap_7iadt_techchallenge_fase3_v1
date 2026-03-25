# Assistente Médico Virtual — IA de Apoio Clínico

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2.0-green?logo=chainlink)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![LLaMA](https://img.shields.io/badge/LLaMA_3-8B_QLoRA-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green?logo=nvidia)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-Acadêmico-lightgrey)

> Sistema de apoio à decisão clínica baseado em LLM fine-tunado com QLoRA, LangChain e LangGraph.
> Desenvolvido como projeto de pós-graduação — FIAP PosTech.

---

## Aviso Importante

**Este sistema é um protótipo acadêmico.** As respostas geradas são baseadas em protocolos
médicos e NÃO substituem a avaliação, diagnóstico ou prescrição de um profissional de saúde
habilitado. Em emergências, ligue **192 (SAMU)**.

---

## Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Fluxo LangGraph](#fluxo-langgraph)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos de Hardware](#pré-requisitos-de-hardware)
- [Instalação dos Drivers NVIDIA e CUDA](#instalação-dos-drivers-nvidia-e-cuda)
- [Instalação do Projeto](#instalação-do-projeto)
- [Como Executar o Fine-tuning](#como-executar-o-fine-tuning)
- [Como Subir a Aplicação](#como-subir-a-aplicação)
- [API REST e Swagger](#api-rest-e-swagger)
- [Interface React (Frontend)](#interface-react-frontend)
- [Como Publicar o Modelo no HuggingFace](#como-publicar-o-modelo-no-huggingface)
- [Avaliação do Modelo](#avaliação-do-modelo)
- [Referência de Comandos (Makefile)](#referência-de-comandos-makefile)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Limitações e Disclaimers](#limitações-e-disclaimers)

---

## Descrição do Projeto

O Assistente Médico Virtual é um sistema completo de inteligência artificial aplicada à
saúde, composto por:

- **Fine-tuning de LLM** com dados médicos usando QLoRA (4-bit quantization) sobre o LLaMA 3 8B
- **RAG (Retrieval-Augmented Generation)** com base de conhecimento indexada no FAISS
- **Orquestração inteligente** com LangChain (chains especializadas) e LangGraph (fluxo automatizado)
- **Interface web React** (Vite + Tailwind CSS) com sidebar e chat moderno, chamando a API REST
- **Módulo de segurança** que valida todas as respostas antes de exibir ao usuário
- **Auditoria completa** de todas as interações em formato JSONL
- **Upload do modelo** para o HuggingFace Hub com Model Card gerado automaticamente

---

## Fluxo LangGraph

```
┌──────────────────────────────────────────────────────────┐
│                   ENTRADA DO PACIENTE                    │
│         (sintomas, exames, histórico, medicamentos)      │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│           NÓ 1: Verificar Exames Pendentes               │
│    Analisa resultados laboratoriais e de imagem          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│           NÓ 2: Consultar Histórico Clínico              │
│    Correlaciona histórico com sintomas atuais            │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│           NÓ 3: Sugerir Conduta / Tratamento             │
│    Gera sugestão baseada em protocolos clínicos          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│             NÓ 4: Validação de Segurança                 │
│    Verifica alertas críticos e aplica regras de safety   │
└──────────────┬───────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │ Há alertas?         │
    │ Sim              Não│
    ▼                     ▼
┌──────────┐     ┌────────────────┐
│  NÓ 5:   │     │    NÓ 6:       │
│ Notificar│────►│ Resposta Final │
│  Equipe  │     │  ao Médico     │
└──────────┘     └────────────────┘
```

---

## Estrutura do Projeto

```
medical-assistant/
│
├── data/
│   ├── raw/                    # Documentos médicos para RAG
│   ├── processed/              # Dataset processado + índice FAISS
│   └── synthetic/              # Exemplos sintéticos gerados
│
├── fine_tuning/
│   ├── prepare_dataset.py      # Pipeline de preparação de dados
│   ├── train.py                # Fine-tuning QLoRA com SFTTrainer
│   ├── evaluate.py             # Avaliação com métricas ROUGE
│   ├── upload_model.py         # Upload do modelo para HuggingFace Hub
│   └── config.yaml             # Hiperparâmetros do treinamento
│
├── assistant/
│   ├── chains.py               # Chains LangChain (qa, exames, tratamento, alerta)
│   ├── graph.py                # Fluxo automatizado LangGraph
│   ├── retriever.py            # RAG com FAISS + SentenceTransformers
│   ├── safety.py               # Validação e limites de atuação
│   └── logger.py               # Auditoria de interações
│
├── interface/
│   └── api.py                  # API REST FastAPI + Swagger UI
│
├── frontend/                   # Interface React (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── App.jsx             # Roteamento entre páginas
│   │   ├── components/
│   │   │   ├── Sidebar.jsx     # Sidebar de navegação escura
│   │   │   ├── Chat.jsx        # Chat clínico com sugestões
│   │   │   ├── Prontuario.jsx  # Formulário LangGraph
│   │   │   ├── Auditoria.jsx   # Tabela de logs + modal de detalhes
│   │   │   └── Sobre.jsx       # Página sobre o sistema
│   │   └── services/
│   │       └── api.js          # Client da FastAPI
│   ├── Dockerfile              # Build Node → nginx
│   ├── nginx.conf              # SPA routing + proxy /api → FastAPI
│   └── package.json
│
├── models/                     # Modelo fine-tunado (gerado após treino)
├── logs/                       # Logs de interações e treino
├── docker/
│   ├── Dockerfile.training     # Container GPU para fine-tuning
│   ├── Dockerfile.app          # Container da aplicação
│   └── nginx.conf              # Proxy reverso (produção)
│
├── docker-compose.yml
├── Makefile                    # Atalhos para todos os comandos
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Pré-requisitos de Hardware

| Uso | GPU mínima | VRAM | RAM |
|---|---|---|---|
| Fine-tuning (QLoRA 4-bit) | NVIDIA RTX 3060 / T4 | 12 GB | 16 GB |
| Fine-tuning (QLoRA 4-bit) ideal | NVIDIA RTX 3090 / A10G | 24 GB | 32 GB |
| Inferência (aplicação) | NVIDIA GTX 1660 / T4 | 6 GB | 8 GB |
| Sem GPU (demo) | — | — | 8 GB |

> Sem GPU: a aplicação carrega automaticamente um modelo menor de demonstração (`microsoft/phi-2`).

---

## Instalação dos Drivers NVIDIA e CUDA

### Windows

**1. Instale o driver NVIDIA:**

Acesse o site oficial e baixe o driver compatível com sua GPU:
`https://www.nvidia.com/drivers`

Ou via PowerShell (detecta e instala automaticamente com winget):
```powershell
winget install NVIDIA.CUDA
```

**2. Verifique a instalação:**
```powershell
nvidia-smi
```
A saída deve mostrar o modelo da GPU, versão do driver e versão CUDA suportada.

**3. Instale o CUDA Toolkit 11.8:**

Baixe em: `https://developer.nvidia.com/cuda-11-8-0-download-archive`

Selecione: Windows → x86_64 → seu Windows → exe (local)

**4. Verifique o CUDA:**
```powershell
nvcc --version
# Deve exibir: Cuda compilation tools, release 11.8
```

---

### Linux (Ubuntu/Debian)

**1. Detecte a GPU e instale o driver recomendado:**
```bash
# Lista drivers disponíveis
ubuntu-drivers devices

# Instala o driver recomendado automaticamente
sudo ubuntu-drivers autoinstall

# Reinicia o sistema
sudo reboot
```

**2. Instale o CUDA Toolkit 11.8:**
```bash
# Adiciona o repositório NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Instala CUDA 11.8
sudo apt-get install -y cuda-11-8

# Adiciona ao PATH (adicione ao ~/.bashrc para persistir)
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**3. Verifique a instalação:**
```bash
nvidia-smi
nvcc --version
```

---

### NVIDIA Container Toolkit (para Docker com GPU)

Necessário para usar GPU dentro dos containers Docker.

**Linux:**
```bash
# Adiciona o repositório
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instala o toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configura o Docker para usar a GPU
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows (WSL2):**

O suporte a GPU via Docker no Windows usa o WSL2. O driver NVIDIA para Windows
já inclui suporte ao WSL2 — basta instalar o driver normal do Windows.

```powershell
# Verifica suporte a GPU no WSL2
wsl nvidia-smi
```

**Teste se o Docker enxerga a GPU:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Instalação do Projeto

### 1. Clone e configure o ambiente

```bash
git clone <url-do-repositorio>
cd medical-assistant

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 2. Configure as variáveis de ambiente

```bash
# Copia o arquivo de exemplo
cp .env.example .env
```

Edite o `.env` com suas credenciais:

```env
# Token HuggingFace com permissão read (e write para upload)
HUGGINGFACE_TOKEN=hf_seu_token_aqui

# Modelo base para fine-tuning
MODEL_BASE=meta-llama/Meta-Llama-3-8B
```

> Crie seu token em: `https://huggingface.co/settings/tokens`
> Para upload do modelo, o token precisa de permissão **write**.

### 3. Verifique se PyTorch reconhece a GPU

```bash
python -c "import torch; print('GPU disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma')"
```

---

## Como Executar o Fine-tuning

### Opção 1: Localmente com GPU

```bash
# Prepara o dataset (baixa PubMedQA + gera sintéticos em português)
make prepare

# Treina o modelo com QLoRA
make train

# Avalia o modelo treinado
make evaluate
```

Ou passo a passo:

```bash
python fine_tuning/prepare_dataset.py
python fine_tuning/train.py
python fine_tuning/evaluate.py --num_samples 20
```

### Opção 2: Via Docker com GPU (recomendado para isolamento)

```bash
# Fine-tuning completo dentro do container
docker compose --profile training up training

# Acompanha os logs
docker compose logs -f training
```

O modelo treinado é salvo em `./models/medical-llama3-qlora/` no host (via volume).

---

## Como Subir a Aplicação

### Localmente

```bash
# API REST + Swagger
make api
# Acesse: http://localhost:8000/docs

# Frontend React (em outro terminal)
make run-frontend
# Acesse: http://localhost:3000
```

### Via Docker — sem GPU (CPU)

```bash
# API + Frontend React
make run-docker

# Ou manualmente
docker compose build api frontend
docker compose up -d api frontend
```

### Via Docker — com GPU (recomendado)

```bash
# API + Frontend React com GPU
make run-docker-gpu

# Ou manualmente
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build api frontend

# Fine-tuning com GPU
make train-docker

# Acompanha os logs
docker compose logs -f api
docker compose logs -f frontend

# Para todos os containers
make stop
```

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **Frontend React** | http://localhost:3000 | Interface principal |
| API REST (Swagger) | http://localhost:8000/docs | Documentação interativa |
| API REST (ReDoc) | http://localhost:8000/redoc | Documentação alternativa |

> **Verificar GPU no Docker:**
> ```bash
> docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
> ```

### Em produção (com nginx)

```bash
docker compose --profile production up -d
# Acesse em: http://localhost (porta 80)
```

---

## API REST e Swagger

A API REST expõe todos os recursos do assistente via HTTP, com documentação interativa automática.

### Iniciar

```bash
# Localmente
make api
# Ou:
PYTHONPATH=. python3 interface/api.py

# Via Docker
docker compose up -d api
```

### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Status da API |
| `GET` | `/stats` | Estatísticas de uso |
| `GET` | `/audit` | Lista log de auditoria (filtros: `limite`, `apenas_alertas`, `sessao`) |
| `GET` | `/audit/{id_interacao}` | Busca interação por ID |
| `POST` | `/chat` | Consulta clínica conversacional |
| `POST` | `/exams` | Análise de exames laboratoriais |
| `POST` | `/treatment` | Sugestão de conduta terapêutica |
| `POST` | `/alert` | Geração de alerta clínico |
| `POST` | `/analyze` | Análise completa via LangGraph |

### Campos obrigatórios por endpoint

Todos os endpoints de `POST` retornam `id_interacao` na resposta, que pode ser usado em `GET /audit/{id_interacao}` para consultar os detalhes completos de auditoria (documentos RAG, scores de similaridade, confiança estimada e motivo da resposta).

| Endpoint | Obrigatório | Opcionais |
|----------|-------------|-----------|
| `POST /chat` | `pergunta` | — |
| `POST /exams` | `exames` | `historico`, `queixas` |
| `POST /treatment` | `diagnostico_hipotetico` | `historico`, `alergias`, `medicamentos_em_uso` |
| `POST /alert` | `dados_paciente`, `achados_criticos` | — |
| `POST /analyze` | `sintomas` | `patient_id`, `exames`, `historico`, `alergias`, `medicamentos` |

### Exemplo de fluxo via API

```bash
# 1. Faz uma pergunta clínica
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Qual o tratamento para HAS estágio 1?"}'

# Resposta: { "id_interacao": "abc123...", "resposta": "...", ... }

# 2. Consulta a auditoria dessa interação
curl http://localhost:8000/audit/abc123...

# Resposta: { documentos_rag, confianca_estimada, motivo_resposta, metadata, ... }
```

---

## Interface React (Frontend)

A interface React é a forma principal de uso. Acesse em `http://localhost:3000` após subir os containers.

### Páginas disponíveis

| Página | Descrição |
|--------|-----------|
| **Consulta Clínica** | Chat com o assistente, sugestões rápidas e aviso de segurança |
| **Prontuário Paciente** | Formulário completo para análise via LangGraph |
| **Log e Auditoria** | Tabela de interações com modal de detalhes (RAG scores, confiança, motivo) |
| **Sobre** | Arquitetura, tecnologias e disclaimers |

### Desenvolvimento local (sem Docker)

```bash
# Instalar dependências
cd frontend
npm install

# Rodar em modo dev (proxia /api → http://localhost:8000)
npm run dev
# Acesse: http://localhost:3000
```

> A FastAPI precisa estar rodando em paralelo: `make api`

---

> As seções abaixo documentam as funcionalidades disponíveis na interface React e via API.

### Consulta Clínica

Chat direto com o assistente médico. Ideal para perguntas clínicas pontuais.

**Como usar:**

1. Digite sua pergunta clínica no campo de texto
2. Pressione **Enter** ou clique em **Enviar Pergunta**
3. O assistente responde com base em protocolos e diretrizes, citando as fontes
4. Faça perguntas de acompanhamento — o contexto da conversa é mantido
5. Clique em **Nova Conversa** para resetar o histórico

**Exemplos de perguntas:**

```
Qual o tratamento de primeira linha para hipertensão arterial estágio 1?
Como diagnosticar diabetes mellitus tipo 2?
Quais são os critérios diagnósticos para sepse?
Como manejar uma crise asmática grave no pronto-socorro?
Quais são as contraindicações ao uso de trombolíticos no AVC?
```

**Exemplo de sessão:**

```
Você:       "Qual o tratamento de primeira linha para HAS estágio 1?"
Assistente: "Para hipertensão estágio 1, o tratamento inicial inclui modificações
             no estilo de vida: redução de sal (< 5g/dia), atividade física aeróbica
             150 min/semana, redução de peso e cessação do tabagismo. Farmacoterapia
             (IECA, BRA, BCC ou diurético tiazídico) é indicada em pacientes com
             risco cardiovascular elevado.
             Fonte: Diretriz Brasileira de Hipertensão 2020."

Você:       "E para estágio 2, muda alguma coisa?"
Assistente: "No estágio 2 (PAS 160-179 ou PAD 100-109 mmHg), a terapia
             farmacológica é recomendada desde o início, geralmente com
             combinação de dois fármacos..."
```

---

### Prontuário / Análise de Paciente

Executa o fluxo completo do **LangGraph** para análise clínica estruturada.
Ideal para avaliação integrada com exames, histórico e medicamentos.

**Como usar:**

1. Preencha o formulário com os dados do paciente:
   - **ID do Paciente** — qualquer identificador (será anonimizado nos logs)
   - **Sintomas e Queixas** *(obrigatório)* — descreva os sintomas, intensidade e tempo
   - **Exames Disponíveis** — resultados de laboratório, ECG, imagem, etc.
   - **Histórico Clínico** — comorbidades, cirurgias, internações anteriores
   - **Alergias** — alergias medicamentosas conhecidas
   - **Medicamentos em Uso** — lista de medicamentos atuais com doses
2. Clique em **Executar Análise LangGraph**
3. Acompanhe o progresso das 6 etapas no painel da direita
4. Verifique os **alertas identificados** e o **relatório final**

**Exemplo de preenchimento:**

```
ID:          PAC-2024-001
Sintomas:    Dor torácica opressiva há 2 horas com irradiação para o braço
             esquerdo, sudorese intensa e dispneia. Início aos esforços.
Exames:      Troponina I: 2.8 ng/mL (VR < 0.04), ECG: supradesnivelamento
             de ST em V1-V4, Hb: 13.2 g/dL, Creatinina: 1.1 mg/dL
Histórico:   HAS há 10 anos, DM2 há 5 anos, tabagista (20 anos/maço).
             Pai com IAM aos 58 anos.
Alergias:    Nenhuma conhecida
Medicamentos: Metformina 850mg 2x/dia, Losartana 50mg 1x/dia
```

**Resultado esperado:**
- Análise dos exames (troponina elevada, padrão ECG)
- Correlação com histórico (fatores de risco cardiovascular)
- Sugestão de conduta (protocolo IAMCSST: ICP primária)
- Alerta CRÍTICO gerado e registrado
- Relatório consolidado com todas as etapas

---

### Log e Auditoria

Exibe um histórico anonimizado das últimas 20 interações com o sistema.

**Como usar:**

1. Clique em **Atualizar Logs** para carregar as interações mais recentes
2. A tabela mostra: timestamp, ID da sessão, chain utilizada, se houve alerta e resumo da pergunta
3. As estatísticas exibem o total de interações e taxa de alertas

> Todos os dados são anonimizados automaticamente antes de serem exibidos.
> CPFs, datas, nomes, e-mails e telefones são substituídos por marcadores.

---

---

## Como Publicar o Modelo no HuggingFace

Após o fine-tuning, publique o modelo para uso futuro ou compartilhamento acadêmico.

**Pré-requisito:** token HuggingFace com permissão **write** no `.env`.

### Repositório padrão (theladycode/NEURIX)

```bash
make upload
# Publica em: https://huggingface.co/theladycode/NEURIX
```

### Repositório alternativo (nome personalizado)

```bash
python fine_tuning/upload_model.py --repo outro-usuario/outro-nome
```

### Repositório privado

```bash
python fine_tuning/upload_model.py --private
```

O script gera automaticamente um **Model Card** profissional com instruções de uso,
informações de treinamento e disclaimers médicos.

### Usar o modelo publicado na aplicação

Atualize o `.env` com o caminho do Hub:

```env
MODEL_PATH=theladycode/NEURIX
```

---

## Avaliação do Modelo

Após o treinamento, o script `evaluate.py` gera métricas automáticas:

| Métrica | Significado | Meta |
|---|---|---|
| ROUGE-1 | Sobreposição de unigramas | > 0.40 |
| ROUGE-2 | Sobreposição de bigramas | > 0.20 |
| ROUGE-L | Subsequência mais longa comum | > 0.35 |

```bash
# Avalia com 50 amostras e salva relatório completo
make evaluate

# Ou com número customizado
python fine_tuning/evaluate.py --num_samples 50

# Relatórios gerados em:
# logs/evaluation_report.json   — JSON com métricas e comparações qualitativas
# logs/evaluation_results.csv   — Tabela para análise em Excel/Pandas
# logs/training_loss.png        — Gráfico da curva de loss do treino
```

---

## Referência de Comandos (Makefile)

```bash
make help          # Lista todos os comandos disponíveis

make setup         # Copia .env.example → .env e instala dependências
make prepare       # Prepara e processa o dataset médico
make train         # Executa o fine-tuning (requer GPU)
make evaluate      # Avalia o modelo com métricas ROUGE
make upload        # Publica o modelo no HuggingFace Hub

make api               # Inicia a API REST + Swagger (http://localhost:8000/docs)
make run-frontend      # Inicia o frontend React em dev (http://localhost:3000)
make run-docker        # Constrói e sobe via Docker sem GPU (API + Frontend React)
make run-docker-gpu    # Constrói e sobe via Docker com GPU (API + Frontend React)
make train-docker      # Fine-tuning via Docker com GPU
make stop              # Para todos os containers Docker

make clean         # Remove cache Python (__pycache__, *.pyc)
```

---

## Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| PyTorch | 2.1.0+ | Framework de deep learning |
| Transformers | 4.40.0+ | Carregamento e inferência de LLMs |
| PEFT | 0.10.0+ | LoRA/QLoRA para fine-tuning eficiente |
| BitsAndBytes | 0.43.0+ | Quantização 4-bit (QLoRA) |
| TRL | 0.8.0+ | SFTTrainer para fine-tuning supervisionado |
| Accelerate | 0.29.0+ | Distribuição e aceleração de treino |
| Datasets | 2.19.0+ | Carregamento de datasets HuggingFace |
| LangChain | 0.2.0+ | Orquestração de chains |
| LangGraph | 0.1.0+ | Fluxos de decisão baseados em grafos |
| FAISS | 1.8.0+ | Busca vetorial eficiente (RAG) |
| SentenceTransformers | 2.7.0+ | Embeddings multilingual |
| React | 18.3+ | Interface web principal (frontend) |
| Vite | 5.4+ | Bundler e dev server do frontend |
| Tailwind CSS | 3.4+ | Estilização do frontend React |
| FastAPI | 0.111.0+ | API REST com Swagger UI automático |
| Uvicorn | 0.29.0+ | Servidor ASGI para a API REST |
| HuggingFace Hub | — | Upload e versionamento de modelos |
| ROUGE Score | 0.1.2+ | Avaliação da qualidade das respostas |
| CUDA | 11.8+ | Aceleração GPU NVIDIA |

---

## Limitações e Disclaimers

1. **Uso acadêmico:** Desenvolvido exclusivamente para fins educacionais — FIAP PosTech
2. **Não certificado:** NÃO possui certificação para uso clínico real (ANVISA, FDA, CE Mark)
3. **Pode errar:** LLMs podem gerar informações incorretas ("alucinações") — sempre valide
4. **Dataset limitado:** Fine-tuning realizado em dataset reduzido por limitações computacionais
5. **Sem diagnóstico definitivo:** O sistema NUNCA faz diagnóstico definitivo, apenas sugestões
6. **Emergências:** Para situações de emergência, acione imediatamente o **SAMU (192)**

---

*Desenvolvido para FIAP PosTech — Pós-graduação em Inteligência Artificial, 2024*
