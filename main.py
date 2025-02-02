import pandas as pd #Manejo de datos en tablas
import re #Expresiones regulares
import torch #Framework para modelos de aprendizaje profundo
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer #Modelos de NLP de hugging face
import chromadb #Base de datos vectorial para busqueda semantica
from fastapi import FastAPI #Framework para apis web rapidas
from pydantic import BaseModel #validacion de datos en fastapi.
from fastapi.middleware.cors import CORSMiddleware #Permite solicitudes de otros dominios
from typing import List #Define listas con tipos en Python


#Inicializamos la api
app = FastAPI()

#Habilita corst en fastapi permitiendo acceso desde cualquier origen 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

casos_legales = pd.read_excel("/home/moi13/prueba_tecnica/sentencias_pasadas.xlsx")

#Renombrar columnas de forma que sean manejables
casos_legales.columns.values[0] = "numero"
casos_legales.columns.values[1] = "relevancia"
casos_legales.columns.values[2] = "providencia"
casos_legales.columns.values[3] = "tipo"
casos_legales.columns.values[4] = "fecha_sentencia"
casos_legales.columns.values[5] = "tema_subtema"
casos_legales.columns = casos_legales.columns.str.strip()

#Si sintencias esta vacia reemplazar por sentencia
casos_legales["sintesis"].fillna(casos_legales["sentencia"], inplace=True)

#Normalizar solicitudes unicas
casos_legales["providencia"] = casos_legales["providencia"].astype(str).str.strip().str.upper()

#Conectar a chroma db
client = chromadb.PersistentClient(path="/home/moi13/prueba_tecnica/chroma_db")

#Reconstruir la base de chroma db cada vez que se ejecute
client.delete_collection(name="casos_legales")
collection = client.get_or_create_collection(name="casos_legales")

#Insertar documentos a chroma db
for _, row in casos_legales.iterrows():
    if row["providencia"]:
        collection.add(
            ids=[row["providencia"]],
            documents=[row["sintesis"]],
            metadatas=[{
                "tipo": row["tipo"],
                "tema": row["tema_subtema"],
                "fecha": str(row["fecha"])
            }]
        )

print("Base de datos en ChromaDB reconstruida correctamente.")
print("IDs almacenados en ChromaDB:", collection.get()["ids"])

#memoria para recordar las ultimas consultas
class MemoriaConversacion:
    def __init__(self): #self representa la instancia de la clase e init inicializa sus atributos al crear el objeto
        self.ultimos_ids = []  
    
    def guardar_ids(self, ids: List[str]): # Guarda los nuevos IDs en la lista
        self.ultimos_ids = ids

    def obtener_ultimos_ids(self):
        return self.ultimos_ids

#Instanciar de manera global la memoria
memoria = MemoriaConversacion()

#Definir estructura de petición
class PreguntaRequest(BaseModel):
    pregunta: str

#Extraer el id del caso con exprsiones regulares
def extraer_ids(pregunta):
    ids_encontrados = re.findall(r"T-\d+/\d+", pregunta.upper())
    print(f"🛠️ IDs extraídos de la pregunta: {ids_encontrados}")
    return ids_encontrados

#Obtener las sentencias desde chroma
def obtener_sentencias(pregunta, recordar=False):
    ids_providencias = extraer_ids(pregunta) #extraer los id de la pregunta
    respuestas = []

    for id_providencia in ids_providencias:
        resultado = collection.get(ids=[id_providencia]) #Busca en la base de datos usando el id

        print(f"Buscando {id_providencia} en chromadb. Resultado: {resultado}")

        if resultado["documents"]:
            sentencia = resultado["documents"][0] #Obtiene el primer documento encontrado
            respuestas.append(f"{id_providencia}: {sentencia.split('.')[0]}.") #Agrega la sentencia a las respuestas 

    if respuestas:
        if recordar:
            memoria.guardar_ids(ids_providencias) #Guarda los id para recordarlos en futuras consultas
        return "\n".join(respuestas) #Devuelve los id en forma de lista

    return "**No encontré información sobre estas sentencias.**"

modelo_path = "/home/moi13/prueba_tecnica/mbart-large-50"
tokenizer = AutoTokenizer.from_pretrained(modelo_path) #Carga el tokenizador del modelo preentrenado
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_path) #Carga el modelo preentrenado (Seq2Seq)

#Obtener contexto relevante desde chroma
def obtener_contexto(pregunta):
    resultados = collection.query(query_texts=[pregunta], n_results=5) #Realiza una consulta en la base de datos collection utilizando la pregunta busqueda obtiene  5 resultados
    if resultados["documents"]:  #Si se encuentran documentos con relacion
        return " ".join(resultados["documents"][0]) #Devuelve los documentos encontrados en un solo string
    return "No hay información relevante en la base de datos."

#Generar respuestas con mbart para preguntas generales
def generar_respuesta_mbart(pregunta):
    contexto = obtener_contexto(pregunta)

    prompt = f"""Explica la respuesta de manera clara, sin tecnicismos, fácil de entender.

    Información clave:
    {contexto}

    Respuesta:"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)# Tokeniza el prompt y lo convierte en tensores para que el modelo pueda procesarlo
    #Parametros para respuesta
    outputs = model.generate(
        **inputs, 
        max_length=250, #Longitud maxima de la respuesta
        temperature=0.7, #Control de aleatoriedad
        top_p=0.9, #Control de diversidad
        num_beams=5,  
        num_return_sequences=1, #Numero de respuestas que el modelo debe generar
        do_sample=True #Utiliza sampling en vez de busqueda deterministica
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True) #Decodifica la salida del modelo a un texto legible
    
    #Limpiar posibles frases tecincas que quedaron
    respuesta = re.sub(r"(Explica la respuesta de manera clara, sin tecnicismos, fácil de entender\.)?", "", respuesta).strip()

    return respuesta

#Generar respuestas del chatbot
def responder_pregunta(pregunta):
    print(f"\n🎯 Pregunta recibida: {pregunta}")  

    #Si es una pregunta de seguimiento se buscan lo ultimos id guardados
    if "de qué se trataron las 3 demandas anteriores" in pregunta.lower():
        ultimos_ids = memoria.obtener_ultimos_ids()
        if not ultimos_ids:
            return "**No tengo memoria de sentencias anteriores. Pregunta por IDs específicos.**"
        
        contexto = obtener_sentencias(", ".join(ultimos_ids))
        return f"Aquí tienes un resumen de las demandas anteriores:\n\n{contexto}"

    #Para consultar sentencias de las demanas
    if "sentencias de las siguientes demandas" in pregunta.lower():
        return obtener_sentencias(pregunta, recordar=True)

    #Usar el mbart si es otra pregunta
    return generar_respuesta_mbart(pregunta)

# 📌 Endpoint API
@app.post("/chat")
def chat_endpoint(request: PreguntaRequest): 
    try:
        respuesta = responder_pregunta(request.pregunta) #Llama a la funcion para obtener la respuesta a la pregunta
        return {"respuesta": respuesta} #La devuelve enformato json
    except Exception as e: #Si no se puede procesar la solicitud
        print(f"Error en chat_endpoint: {e}") 
        return {"error": "Hubo un problema al procesar la solicitud."}

#Mensaje del servidor
@app.get("/")
def read_root():
    return {"mensaje": "Servidor funcionando correctamente 🎯"}
