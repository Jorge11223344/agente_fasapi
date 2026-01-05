"""
FastAPI - Agente de Ventas de Arena Sanitaria
API REST moderna para aprender FastAPI con un caso real
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai  # ← IMPORT CORREGIDO

# Cargar variables de entorno
load_dotenv()

# Configuración de la app FastAPI
app = FastAPI(
    title="Arenito - API de Ventas",
    description="API REST para agente de ventas de arena sanitaria",
    version="1.0.0"
)

# Configurar CORS (permitir peticiones desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS PYDANTIC ====================
# Pydantic valida automáticamente los datos de entrada/salida

class Message(BaseModel):
    """Modelo para un mensaje individual en el historial"""
    role: str = Field(..., description="Rol del mensaje: 'user' o 'model'")
    text: str = Field(..., description="Contenido del mensaje")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "text": "Hola, necesito arena para mi gato",
                "timestamp": "2024-01-04T10:30:00"
            }
        }


class ChatRequest(BaseModel):
    """Modelo para la petición de chat"""
    message: str = Field(..., min_length=1, description="Mensaje del usuario")
    history: List[Message] = Field(default=[], description="Historial de conversación")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Cuánto cuesta 20kg?",
                "history": [
                    {"role": "user", "text": "Hola"},
                    {"role": "model", "text": "¡Hola! ¿En qué puedo ayudarte?"}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Modelo para la respuesta del chat"""
    answer: str = Field(..., description="Respuesta del agente")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Los 20kg cuestan $13.890 CLP",
                "timestamp": "2024-01-04T10:30:05"
            }
        }


class Product(BaseModel):
    """Modelo para productos del catálogo"""
    weight_kg: int = Field(..., description="Peso en kilogramos")
    price_clp: int = Field(..., description="Precio en CLP")
    description: str = Field(..., description="Descripción del formato")


class HealthCheck(BaseModel):
    """Modelo para verificar el estado de la API"""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== CONFIGURACIÓN DEL AGENTE ====================

SYSTEM_INSTRUCTION = """
Eres un agente de ventas para una tienda chilena de arena sanitaria de bentonita con aroma lavanda.
Tu moneda es CLP (IVA incluido).

REGLAS CRÍTICAS DE MEMORIA:
- SIEMPRE recuerda TODO lo que el cliente te ha dicho en esta conversación
- Si el cliente ya te dio su nombre, dirección o preferencias, NO vuelvas a preguntarlo
- Si ya cotizaste algo, recuerda los detalles exactos
- Mantén consistencia total con información previa de la conversación

OBJETIVO PRINCIPAL  
Saludar amablemente, responder a la consulta que el cliente pregunte, mostrar claramente la gama de precios disponible y guiarlo hacia la compra de forma breve, cercana y profesional.

FLUJO DE CONVERSACIÓN OBLIGATORIO  
1) Saluda cordialmente y preséntate SOLO en el primer mensaje

3) Muestra la gama de formatos y precios disponibles cuando sea relevante
4) Si el cliente indica interés, indica el precio de lo que requiere.
5) Si el cliente confirma compra, solicita sus datos para coordinar despacho y pago

CATÁLOGO DE PRECIOS (MEMORIZA ESTOS PRECIOS - NO LOS OLVIDES) :
• 8 kg: $6.490  
• 16 kg: $11.990  
• 20 kg: $13.890  
• 24 kg: $16.990  
• 28 kg: $18.990  
• 32 kg: $21.990  
• 40 kg: $26.990  (pueden ser 5 bolsas de 8 kgs o 2 bolsas de 20 kgs)

el cliente podría indicarte el primer numero como identificador de la promoción que necesita, ejemplo si te coloca 8 quiere decir que te esta solicitando la promoción de  8 kg: $6.490  (una bolsa de 8 kg).

REGLAS DE COTIZACIÓN  
- Siempre mostrar precios en CLP con separador de miles 
- NUNCA cambies los precios una vez cotizados
- Cotizar con desglose por línea (formato × precio = subtotal)
- Mostrar el TOTAL al final
- No inventar descuentos, packs ni productos que no estén en la lista
- No mencionar precios antiguos ni catálogos distintos
- Si ya cotizaste algo, mantén el mismo precio

DESPACHO (INFORMACIÓN IMPORTANTE - NO OLVIDES):
- Despachos disponibles en: San Pedro de la Paz, Higeras, Hualpén y Concepción
- Para Talcahuano: solo atendemos hasta sectores como Salinas o Gaete (envío gratis)
- Envío GRATIS por compras sobre $10.000
- Si preguntan por otras comunas, indicar que el costo de envío debe confirmarse y parte en $1.500

CIERRE DE VENTA  
Si el cliente confirma que desea comprar, solicita de forma amable, los siguientes datos preguntalos uno a la vez, no todos al mismo tiempo:
• Nombre (basta con su primer nombre)
• Teléfono con el que se coordinaría la entrega
• Dirección incluyendo la comuna, para chequear si corresponde a las comunas con despacho gratis. En caso que la comuna sea diferente indicarle que tendrá un recargo por despacho.
• Forma de pago: efectivo, transferencia o link de pago

Si elige TRANSFERENCIA, proporciona estos datos:
Nombre: JIMACOMEX SpA
RUT: 78.146.748-0
Banco de Chile
Cuenta vista: 00-011-06251-91
Email: jimunozacuna@gmail.com

Pueden depositar antes o al momento de la entrega, sin problemas.

• Horario para recibir: 
  - Ventana preferida, si el cliente te coloca 1 corresonde al horario entre 15 y 17 horas, si te coloca 2 es despues de las 20 horas y si te coloca 3 debe esperar coordinación con nosotros y nos debe indicar el horario que puede recibir en un determinado rango..:
    1.- De 15:00 a 17:00 hrs
    2.- Después de las 20:00 hrs
    3.- Si el cliente prefiere otro horario, coordinamos según su disponibilidad

TONO Y ESTILO  
- Cercano, respetuoso y profesional
- Respuestas breves, claras y enfocadas en vender
- No usar lenguaje técnico innecesario
- Siempre ofrecer ayuda adicional al final de cada respuesta
- MANTÉN CONSISTENCIA: si ya dijiste algo, no te contradices

CONFIRMACIÓN FINAL DE COMPRA:
Al termino de la conversación le debes solicitar que confirme si los datos registrados son los que pidió, Resume: kilos, valor total, horario de entrega, dirección y agradece al cliente.
"""

# Catálogo de productos (opcional, para consultas directas)
CATALOG = [
    Product(weight_kg=8, price_clp=6490, description="1 bolsa de 8 kg"),
    Product(weight_kg=16, price_clp=11990, description="2 bolsas de 8 kg"),
    Product(weight_kg=20, price_clp=13890, description="1 bolsa de 20 kg"),
    Product(weight_kg=24, price_clp=16990, description="3 bolsas de 8 kg"),
    Product(weight_kg=28, price_clp=18990, description="1 bolsa de 20 kg + 1 bolsa de 8 kg"),
    Product(weight_kg=32, price_clp=21990, description="4 bolsas de 8 kg"),
    Product(weight_kg=40, price_clp=26990, description="5 bolsas de 8 kg o 2 bolsas de 20 kg"),
]


# ==================== ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def home():
    """
    Endpoint principal que muestra la interfaz web del chat
    """
    return """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Arenito - Agente de Ventas FastAPI</title>
</head>
<body style="font-family: Arial; background:#f6f6f6;">
  <div style="max-width:800px;margin:30px auto;background:#fff;padding:18px;border-radius:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <h2 style="margin:0;">Hola, soy Arenito 🐾 (FastAPI Version)</h2>
      <button id="btnReset" style="padding:8px 12px;border-radius:8px;background:#ff6b6b;color:white;border:none;cursor:pointer;">
        🔄 Nueva Conversación
      </button>
    </div>
    <div id="history" style="height:55vh;overflow:auto;border:1px solid #eee;padding:12px;border-radius:10px;"></div>

    <form id="form" style="display:flex;gap:10px;margin-top:10px;">
      <textarea id="input" rows="1" style="flex:1;padding:10px;border-radius:10px;"></textarea>
      <button id="btn" type="submit" style="padding:10px 14px;border-radius:10px;">Enviar</button>
    </form>
  </div>

<script>
  const historyEl = document.getElementById("history");
  const form = document.getElementById("form");
  const input = document.getElementById("input");
  const btn = document.getElementById("btn");
  const btnReset = document.getElementById("btnReset");

  let history = [];

  function add(role, text) {
    const div = document.createElement("div");
    div.style.margin = "10px 0";
    div.style.padding = "10px 12px";
    div.style.borderRadius = "10px";
    div.style.maxWidth = "85%";
    div.style.whiteSpace = "pre-wrap";
    div.style.background = role === "user" ? "#e8f0ff" : "#f2f2f2";
    if (role === "user") div.style.marginLeft = "auto";
    div.textContent = text;
    historyEl.appendChild(div);
    historyEl.scrollTop = historyEl.scrollHeight;

    history.push({ role: role === "user" ? "user" : "model", text });
    // Mantener más historial en memoria (60 mensajes en lugar de 12)
    if (history.length > 60) history = history.slice(-60);
  }

  function resetConversation() {
    if (confirm("¿Estás seguro de iniciar una nueva conversación? Se perderá el historial actual.")) {
      history = [];
      historyEl.innerHTML = "";
      add("bot", "¡Hola! Soy tu asistente de ventas para arena sanitaria. ¿En qué puedo ayudarte hoy?");
    }
  }

  // Evento para el botón de reset
  btnReset.addEventListener("click", resetConversation);

  // Inicializar conversación
  add("bot", "¡Hola! Soy tu asistente de ventas para arena sanitaria. ¿En qué puedo ayudarte hoy?");

  async function send(message) {
    btn.disabled = true; input.disabled = true;

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history })
    });

    const data = await res.json().catch(() => ({}));
    btn.disabled = false; input.disabled = false; input.focus();

    if (!res.ok) return add("bot", data.detail || "Ocurrió un error.");
    add("bot", data.answer || "");
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";
    add("user", msg);
    send(msg);
  });
</script>
</body>
</html>
"""


@app.get("/health", response_model=HealthCheck, tags=["Sistema"])
async def health_check():
    """
    Verifica que la API esté funcionando correctamente
    """
    return HealthCheck(
        status="ok",
        message="API funcionando correctamente"
    )


@app.get("/api/catalog", response_model=List[Product], tags=["Productos"])
async def get_catalog():
    """
    Retorna el catálogo completo de productos disponibles
    """
    return CATALOG


@app.get("/api/catalog/{weight_kg}", response_model=Product, tags=["Productos"])
async def get_product(weight_kg: int):
    """
    Obtiene información de un producto específico por peso
    """
    product = next((p for p in CATALOG if p.weight_kg == weight_kg), None)
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    return product


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Endpoint principal del chat que procesa mensajes y retorna respuestas del agente
    
    - **message**: Mensaje del usuario
    - **history**: Historial de la conversación (opcional)
    """
    
    # Validar que existe la API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="Falta configurar GEMINI_API_KEY en las variables de entorno"
        )
    
    # Configurar Gemini con la API key
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={
                "temperature": 0.7,  # Más bajo = más consistente y predecible
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al configurar Gemini: {str(e)}")
    
    # Preparar el historial para Gemini (últimos 40 mensajes para mejor contexto)
    # Convertir el historial al formato que espera Gemini
    chat_history = []
    for msg in request.history[-40:]:  # Aumentado de 12 a 40 mensajes
        if msg.role in ("user", "model") and msg.text:
            chat_history.append({
                "role": msg.role,
                "parts": [msg.text]  # Gemini espera directamente el texto
            })
    
    # Iniciar el chat con historial
    try:
        chat = model.start_chat(history=chat_history)
        # Enviar el mensaje actual
        response = chat.send_message(request.message)
        
        answer = (response.text or "").strip()
        
        if not answer:
            raise HTTPException(status_code=500, detail="Gemini no generó respuesta")
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar con Gemini: {str(e)}")


# ==================== PUNTO DE ENTRADA ====================

if __name__ == "__main__":
    import uvicorn
    
    # Ejecutar el servidor en modo desarrollo
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload cuando cambies código
    )