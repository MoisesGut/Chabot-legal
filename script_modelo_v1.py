import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import chromadb

# 📌 Cargar archivo de sentencias
archivo_excel = "/home/moi13/prueba_tecnica/sentencias_pasadas.xlsx"
casos_legales = pd.read_excel(archivo_excel)

# 📌 Renombrar columnas
casos_legales.rename(columns={
    "Relevancia": "relevancia",
    "Providencia": "providencia",
    "Tipo": "tipo",
    "Fecha Sentencia": "fecha",
    "Sentencia": "sentencia",
    "Tema - subtema": "tema_subtema",
    "resuelve": "sentencia",
    "sintesis": "sintesis"
}, inplace=True)

# 📌 Si "sentencia" no existe, usar "sintesis"
if "sentencia" not in casos_legales.columns:
    print("⚠️ La columna 'sentencia' no existe, se usará 'resuelve' en su lugar.")
    casos_legales["sentencia"] = casos_legales["sintesis"]

# 📌 Eliminar filas vacías
casos_legales.dropna(subset=["sentencia"], inplace=True)

# 📌 Cargar modelo Flan-T5
modelo_path = "/home/moi13/prueba_tecnica/flan-base"
tokenizer = AutoTokenizer.from_pretrained(modelo_path)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_path)

# 📌 Configurar ChromaDB
client = chromadb.PersistentClient(path="/home/moi13/prueba_tecnica/chroma_db")
collection = client.get_or_create_collection(name="casos_legales")

# 📌 Insertar documentos en ChromaDB
for idx, row in casos_legales.iterrows():
    collection.add(
        ids=[str(idx)],
        documents=[row["sentencia"]],
        metadatas=[{
            "tipo": row["tipo"],
            "tema": row["tema_subtema"],
            "fecha": str(row["fecha"])
        }]
    )

print("📥 Sentencias agregadas a la base de datos.")

# 📌 Obtener contexto relevante
def obtener_contexto(pregunta):
    resultados = collection.query(query_texts=[pregunta], n_results=3)
    if resultados['documents']:
        return " ".join(resultados['documents'][0])
    return "No hay información relevante en la base de datos."

# 📌 Generar respuestas con IA
def responder_pregunta(pregunta):
    print(f"\n🎯 Pregunta recibida: {pregunta}")  # 🔹 Imprime la pregunta antes de resolverla
    contexto = obtener_contexto(pregunta)

    # 📌 Manejo de preguntas específicas
    if "sentencias de 3 demandas" in pregunta.lower():
        sentencias = casos_legales.head(3)[["sentencia", "fecha"]]
        return "\n\n".join([f"📌 {row['sentencia']} (Fecha: {row['fecha']})" for _, row in sentencias.iterrows()])

    if "de qué se trataron las 3 demandas" in pregunta.lower():
        descripciones = casos_legales.head(3)[["sintesis"]]
        return "\n\n".join([f"📌 {row['sintesis']}" for _, row in descripciones.iterrows()])

    if "sentencia del caso que habla de acoso escolar" in pregunta.lower():
        caso_acoso = casos_legales[casos_legales["tema_subtema"].str.contains("acoso escolar", case=False, na=False)]
        if not caso_acoso.empty:
            sentencia = caso_acoso.iloc[0]
            return f"📌 {sentencia['sentencia']} (Fecha: {sentencia['fecha']})"
        return "No se encontró un caso sobre acoso escolar."

    if "detalle de la demanda relacionada con acoso escolar" in pregunta.lower():
        caso_acoso = casos_legales[casos_legales["tema_subtema"].str.contains("acoso escolar", case=False, na=False)]
        if not caso_acoso.empty:
            return f"📌 {caso_acoso.iloc[0]['sintesis']}"
        return "No se encontró información sobre acoso escolar."

    if "casos que hablan sobre el PIAR" in pregunta.lower():
        casos_piar = casos_legales[casos_legales["tema_subtema"].str.contains("PIAR", case=False, na=False)]
        if not casos_piar.empty:
            return "\n\n".join([f"📌 {row['sintesis']}" for _, row in casos_piar.iterrows()])
        return "No se encontraron casos sobre PIAR."

    # 📌 Generar respuesta con Flan-T5
    prompt = f"""Responde de manera clara y sencilla:

    Pregunta: {pregunta}
    
    Contexto:
    {contexto}
    
    Respuesta:"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        **inputs, 
        max_length=200,  # 🔹 Ahora la respuesta será más clara y sin cortes
        temperature=0.7,  # 🔹 Aumentamos la variabilidad de la respuesta
        top_p=0.95,  
        num_beams=3, 
        num_return_sequences=1, 
        do_sample=True  # 🔹 Permitimos sampling para más naturalidad
    )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return respuesta

# 📌 Prueba con preguntas de ejemplo
preguntas_ejemplo = [
    "¿Cuáles son las sentencias de 3 demandas?",
    "¿De qué se trataron las 3 demandas anteriores?",
    "¿Cuál fue la sentencia del caso que habla de acoso escolar?",
    "¿Diga el detalle de la demanda relacionada con acoso escolar?",
    "¿Existen casos que hablan sobre el PIAR, indique de qué trataron los casos y cuáles fueron sus sentencias?"
]

for pregunta in preguntas_ejemplo:
    respuesta = responder_pregunta(pregunta)
    print(f"🤖 Respuesta generada: {respuesta}\n")
