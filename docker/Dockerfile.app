# ============================================================
# Dockerfile.app
# Container da aplicação (assistente médico + interface Gradio)
# Usa Python slim para imagem menor em produção
# ============================================================

FROM python:3.11-slim

# Metadados do container
LABEL maintainer="FIAP PosTech"
LABEL description="Container da aplicação do assistente médico (Gradio + LangChain)"
LABEL version="1.0.0"

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências de sistema mínimas necessárias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python (camada de cache otimizada)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia os módulos da aplicação
COPY assistant/ ./assistant/
COPY interface/ ./interface/

# Cria os diretórios necessários (conteúdo montado via volumes)
RUN mkdir -p models logs data/raw data/processed data/synthetic

# Expõe a porta do Gradio
EXPOSE 7860

# Variáveis de ambiente padrão
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV GRADIO_SHARE=false
ENV APP_PORT=7860

# Health check para verificar se a aplicação está rodando
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Comando para iniciar a interface Gradio
CMD ["python", "interface/app.py"]
