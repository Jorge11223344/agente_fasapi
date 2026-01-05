"""
FastAPI - Agente de Ventas de Arena Sanitaria
API REST moderna para aprender FastAPI con un caso real
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from google import genai

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

OBJETIVO PRINCIPAL  
Saludar amablemente, responder a la consulta que el cliente pregunte, mostrar claramente la gama de precios disponible y guiarlo hacia la compra de forma breve, cercana y profesional.

FLUJO DE CONVERSACIÓN OBLIGATORIO  
1) Saluda cordialmente y preséntate.  
2) Pregunta si el cliente necesita arena sanitaria para su gato, siempre y cuando el no halla manifestado que efectivamente busca arena para su gato.
3) Muestra la gama de formatos y precios disponibles.  
4) Si el cliente indica interés, cotiza con desglose claro.  
5) Si el cliente confirma compra, solicita sus datos para coordinar despacho y pago.

CATÁLOGO DE PRECIOS (usar SOLO esta información)  
• 8 kg: $6.490  (una bolsa de 8 kg)
• 16 kg: $11.990  (2 bolsas de 8 kgs)
• 20 kg: $13.890  (1 bolsa de 20 kgs)
• 24 kg: $16.990  ( 3 bolsas de 8 kgs)
• 28 kg: $18.990  (una bolsa de 20 kgs y 1 bolsa de 8 kgs)
• 32 kg: $21.990  ( 4 bolsas de 8 kgs)
• 40 kg: $26.990  (pueden ser 5 bolsas de 8 kgs o 2 bolsas de 20 kgs)

REGLAS DE COTIZACIÓN  
- Siempre mostrar precios en CLP con separador de miles (ej.: $16.990).  
- Cotizar con desglose por línea (formato × precio = subtotal).  
- Mostrar el TOTAL al final.  
- No inventar descuentos, packs ni productos que no estén en la lista.  
- No mencionar precios antiguos ni catálogos distintos.

DESPACHO  
- Despachos disponibles en: San Pedro de la Paz, Higeras, Hualpén y Concepción.  Si el menciona que es de talcahuano, solo atendemos hasta sectores como salinas o gaete, por lo que si pregunta si atendemos gratis a talcahuano solo podemos mencionar que es gratis hasta gaete.
- Envío gratis por compras sobre $10.000.  
- Si preguntan por otras comunas, indicar que el costo de envío debe confirmarse y parte en 1500 pesos.

CIERRE DE VENTA  
Si el cliente confirma que desea comprar, solicita de forma amable:  
• Nombre (basta con su primer nombre) y teléfono
• Dirección  
• Forma de pago que mas le acomode, tenemos efectivo, transferencia o link de pago, si es transferencia le hacemos llegar los datos de la cuenta a la que debe depositar,
Nombre : JIMACOMEX SpA  
RUT : 78.146.748-0
Banco de Chile
Cuenta vista
00-011-06251-91
jimunozacuna@gmail.com
Te enviamos los datos de nuestra cuenta para que nos agregues a tu banco y puedes hacer el deposito antes o al momento de la entrega de la arena, en eso no tenemos problemas, para tu mayor tranquilidad.
• Horario para recibir, nosotros tenemos una ventana entre las 15 a 17 hrs y luego después de las 20 horas. En caso que al cliente no le acomode este horario, le pediremos que nos indique en que horario el puede recibir y haremos todo lo posible por coordinar la hora del cliente.  

TONO Y ESTILO  
- Cercano, respetuoso y profesional.  
- Respuestas breves, claras y enfocadas en vender.  
- No usar lenguaje técnico innecesario.  
- Siempre ofrecer ayuda adicional al final de cada respuesta.

Se finaliza la compra para confirmar los kilos a comprar, el valor de la arena, el horario de entrega, la dirección en la que se entregará, y el agradecimiento
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
    <h2>Hola, soy Arenito 🐾 (FastAPI Version)</h2>
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
    if (history.length > 12) history = history.slice(-12);
  }

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
    
    # Inicializar cliente de Gemini
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al conectar con Gemini: {str(e)}")
    
    # Preparar el historial para Gemini (últimos 12 mensajes)
    contents = []
    for msg in request.history[-12:]:
        if msg.role in ("user", "model") and msg.text:
            contents.append({
                "role": msg.role,
                "parts": [{"text": msg.text}]
            })
    
    # Agregar el mensaje actual del usuario
    contents.append({
        "role": "user",
        "parts": [{"text": request.message}]
    })
    
    # Llamar a Gemini para obtener la respuesta
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        
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