# ============================================================
# Makefile — Comandos de conveniência para o projeto
# ============================================================
# Uso: make <comando>
# Exemplo: make setup, make train, make run

.PHONY: help setup prepare train evaluate upload run run-docker stop clean test

# Exibe ajuda com todos os comandos disponíveis
help:
	@echo ""
	@echo "Assistente Medico Virtual — Comandos disponíveis:"
	@echo "=================================================="
	@echo "  make test        Roda todos os testes dos módulos"
	@echo "  make setup       Cria .env e instala dependências"
	@echo "  make prepare     Prepara e processa o dataset"
	@echo "  make train       Executa o fine-tuning do modelo"
	@echo "  make evaluate    Avalia o modelo treinado"
	@echo "  make run         Inicia a interface Gradio localmente"
	@echo "  make run-frontend    Inicia o frontend React em dev (http://localhost:3000)"
	@echo "  make run-docker      Inicia API + Frontend via Docker (CPU)"
	@echo "  make run-docker-gpu  Inicia API + Frontend via Docker (GPU)"
	@echo "  make train-docker Executa o fine-tuning via Docker"
	@echo "  make upload      Faz upload do modelo para o HuggingFace Hub"
	@echo "  make stop        Para os containers Docker"
	@echo "  make clean       Remove cache e arquivos temporários"
	@echo ""

# Roda todos os testes dos módulos
test:
	@echo "Executando testes dos módulos..."
	python tests/test_modules.py

# Configura o ambiente de desenvolvimento
setup:
	@echo "Copiando .env.example para .env..."
	@cp -n .env.example .env || echo ".env ja existe, pulando."
	@echo "Instalando dependências Python..."
	pip install -r requirements.txt
	@echo "Pronto! Edite o arquivo .env com suas credenciais."

# Prepara o dataset médico
prepare:
	@echo "Preparando dataset medico..."
	python fine_tuning/prepare_dataset.py

# Executa o fine-tuning localmente (requer GPU)
train: prepare
	@echo "Iniciando fine-tuning com QLoRA..."
	python fine_tuning/train.py

# Avalia o modelo treinado
evaluate:
	@echo "Avaliando modelo com metricas ROUGE..."
	python fine_tuning/evaluate.py --num_samples 20

# Faz upload do modelo para o HuggingFace Hub
upload:
	@echo "Fazendo upload do modelo para o HuggingFace Hub..."
	python fine_tuning/upload_model.py

# Inicia a interface Gradio localmente
run:
	@echo "Iniciando interface Gradio em http://localhost:7860"
	python interface/app.py

# Inicia a API REST com Swagger
api:
	@echo "Iniciando API REST em http://localhost:8000/docs"
	PYTHONPATH=. python interface/api.py

# Inicia o frontend React em modo desenvolvimento (requer Node.js)
run-frontend:
	@echo "Iniciando frontend React em http://localhost:3000"
	cd frontend && npm install && npm run dev

# Constrói e inicia API + Frontend React via Docker — sem GPU
run-docker:
	@echo "Construindo e iniciando containers Docker (CPU)..."
	docker compose build api frontend
	docker compose up api frontend

# Constrói e inicia API + Frontend React via Docker — com GPU (requer NVIDIA Container Toolkit)
run-docker-gpu:
	@echo "Construindo e iniciando containers Docker (GPU)..."
	docker compose build api frontend
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api frontend

# Executa o fine-tuning via Docker (requer NVIDIA Container Toolkit)
train-docker:
	@echo "Executando fine-tuning via Docker (requer GPU)..."
	docker compose --profile training up training

# Para todos os containers
stop:
	docker compose down

# Remove arquivos de cache Python
clean:
	@echo "Removendo cache Python..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Pronto."
