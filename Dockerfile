# Usa l'immagine base Python
FROM python:3.11-slim

# Crea una directory di lavoro
WORKDIR /app

# Copia i file della tua app
COPY . /app

# Installa le dipendenze (se ci sono)
RUN pip install --no-cache-dir -r requirements.txt || true

# Comando da eseguire all'avvio
CMD ["python", "test-bot.py"]
