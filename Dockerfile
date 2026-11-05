# Imagen base ligera de Python
FROM python:3.10-slim

# Evitar la creación de archivos .pyc y forzar logs en tiempo real
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación y el modelo entrenado
COPY main.py .
COPY best_convnext_categories.pth .

# Puertos expuestos
EXPOSE 8000

# Comando para arrancar el servidor Uvicorn en producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]