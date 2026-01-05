# agente_fasapi# Arenito FastAPI

Agente de ventas de arena sanitaria con FastAPI y Google Gemini.

## Instalación

1. Crear entorno virtual:
```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
```

2. Instalar dependencias:
```bash
   pip install -r requirements.txt
```

3. Configurar .env:
```bash
   cp .env.example .env
   # Editar .env con tu API key
```

4. Ejecutar:
```bash
   python main.py
```

5. Abrir: http://localhost:8000/docs