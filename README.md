# Índice

- [Chatbot de Sentencias Judiciales 📜](#chatbot-de-sentencias-judiciales-)
  - [Características principales](#características-principales)
  - [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Instalación y configuración](#instalación-y-configuración)
- [PRUEBA TÉCNICA](#prueba-técnica)
- [Investigación](#investigación)
- [Desarrollo](#desarrollo)
  -[Preguntas](#preguntas)
- [Informe](#informe)
- [Supuestos](#supuestos)
- [Formas para resolver el caso y la opción tomada en esta prueba](#formas-para-resolver-el-caso-y-la-opción-tomada-en-esta-prueba)
- [Resultados del análisis de datos y de lo modelos](#resultados-del-análisis-de-datos-y-de-lo-modelos)
  - [Resultado Análisis de datos](#resultado-análisis-de-datos)
  - [Análisis de sentencias por mes](#análisis-de-sentencias-por-mes)
  - [Resultado del análisis de los temas más comunes](#resultado-del-análisis-de-los-temas-más-comunes)
  - [Análisis de la Nube de Palabras de los temas y síntesis de los casos](#análisis-de-la-nube-de-palabras-de-los-temas-y-síntesis-de-los-casos)
- [Resultados del Análisis de los Modelos](#resultados-del-análisis-de-los-modelos)
  - [Evaluación del Rendimiento del Sistema](#evaluación-del-rendimiento-del-sistema)
  - [Análisis de la Precisión y Calidad de las Respuestas](#análisis-de-la-precisión-y-calidad-de-las-respuestas)
- [Futuros Ajustes o Mejoras](#futuros-ajustes-o-mejoras)
- [Apreciaciones y Comentarios del Caso](#apreciaciones-y-comentarios-del-caso)
- [Conclusión](#conclusión)
- [Referencias](#referencias)

# Chatbot de Sentencias Judiciales 📜  

Este chatbot está diseñado para responder preguntas sobre sentencias judiciales utilizando **FastAPI, ChromaDB y un modelo de lenguaje natural basado en transformers**. Su objetivo es proporcionar respuestas en **lenguaje claro y comprensible** para la mayoría de las personas, sin tecnicismos legales complejos.

---

## Características principales  
* **Búsqueda de sentencias por ID**  
* **Respuestas en lenguaje natural, sin tecnicismos**  
* **Memoria conversacional**: recuerda sentencias consultadas recientemente  
* **Base de datos vectorial con ChromaDB** para respuestas más precisas  
* **Interfaz de usuario interactiva con React**  
* **Exposición del backend con Ngrok** para pruebas remotas  

---

## Tecnologías utilizadas  

| Tecnología | Descripción |
|------------|------------|
| 🐍 **Python** | Lenguaje de programación del backend |
| ⚡ **FastAPI** | Framework para crear APIs rápidas |
| 🔍 **ChromaDB** | Base de datos vectorial para búsqueda eficiente |
| 🤖 **Transformers (Hugging Face)** | Modelo mBART para respuestas en lenguaje natural |
| 🎨 **React** | Framework de frontend para la interfaz del chatbot |
| 🌍 **Ngrok** | Herramienta para exponer el backend de manera segura |

---

## Instalación y configuración  

### **IMPORTANTE**  
El presente reporitorio se uso para hacer una prueba tecnica en azure, por obvias razones no se comparte bases de datos, para compilar el codigo main.py es necesario tener descargado el modelo de IA BART-BASE-5 El cual esta disponible en el repositorio oficial de hugging face o puedes descargarlo directamente desde de tu codigo, además se tienen que instalar las librerias necesariasdisponibles en requeriments.txt, el proyecto consiste en el desarrollo de un back end en python "main.py", un desarrollo de front end "App.js" y "App.css" sin embargo si se cuenta con un equipo de computo con por lo menos 16gb ram y gpu puedes correr el modelo directamente desde tu maquina local usando el codigo "modelo.py", así mismo se realizo un analisis de los datos de la base "Analisis_Sentencias.ipynb" disponible en el repositorio actual.

:warning: Si se desea hacer pruebas de la aplicacion ingresar al siguiente link, la disponibilidad esta sujeta al servidor el cual se encuentra en una maquina virtual: chatbot-legal.ngrok.app :warning:
 
---

# PRUEBA TÉCNICA

**Caso de Negocio:** Asesor Legal para Consulta de Historia de Demandas

## Informe de Resultados

---

**Elaborado por:**  
Gutierrez Carmona Moisés

---

**Fecha de Entrega:** 04 de febrero de 2025

# Investigación

**Modelos de IA Generativa**

Este tipo de modelos, sobre todo los basados en arquitecturas de Transformers, han sido importantes en el campo del Procesamiento del Lenguaje Natural. Estos modelos se entrenan a partir de grandes cantidades de datos y son capaces de generar respuestas coherentes y con contexto.

Mbart es una extensión del modelo BART que se ha desarrollado en trabajar en un contexto multilingüe. La versión usada en esta prueba denominada mBart-BASE-5 representa una variante optimizada para 5 idiomas, permitiendo generar textos en múltiples lenguajes con una alta calidad. Este modelo utiliza un proceso de preentrenamiento mediante eliminación de ruido de los datos que permite aprender representaciones lingüísticas robustas. La elección de este modelo para la prueba fue debido a que uno de los 5 idiomas en los que se especializa es el español y es un modelo bastante ligero que no requiere mucha capacidad de computo. [1]

**Bases vectoriales** 

Son estructuras de datos que permiten el almacenamiento y la búsqueda de información a partir de la representación numérica de los datos (vectores). Estas bases facilitan la búsqueda semántica.

Para esta prueba se uso Chroma DB, la cual esta diseñada para integrarse con modelos de embedding. Permitiendo indexar y recuperar información de forma eficiente, lo que hace eficiente la búsqueda semántica rápida y escalable. [3] 

**Limitaciones de un chabot.**

Dentro de esta prueba técnica se percibieron las siguientes limitaciones:

Aunque los modelos han mejorado, comprender contextos complejos siguen siendo un desafío. Los chatbots pueden perder el hilo de la conversación. Los sesgos y limitaciones de los datos de entrenamiento pueden llevar a respuestas inadecuadas o parciales. En los entornos que son constantemente dinámicos o con información muy especializada, los chatbots pueden carecer de la capacidad de adaptarse a preguntas nuevas sin un reentrenamiento. [2]

# Desarrollo

La siguiente prueba técnica se llevó a cabo en **Azure**, utilizando los siguientes recursos:  

- **Máquina virtual (Standard D4as v6, 4 vCPU, 16 GiB de memoria):** Seleccionada para garantizar un rendimiento óptimo del modelo de IA generativa, aprovechando su capacidad de cómputo para procesar consultas de manera eficiente.
  ![image](https://github.com/user-attachments/assets/c46dfe9a-2e0c-406d-ba4c-22e4434ddfe3)
-	**File storage (Recursos Compartdios):):** Con la finalidad de compartir recursos con la maquina virtual, entre los que se encuentra los archivos necesario para el modelo de IA generativa.
  ![image](https://github.com/user-attachments/assets/c88ab392-98c6-43ca-9d4e-cf1d67f68a20)
- Se creó un grupo de recursos para contene todo lo que ocuparemos.
 	![image](https://github.com/user-attachments/assets/ad90963b-d167-476d-a60f-2c3b88f8e4ec)
- Se creó un entorno optimizado para garantizar su correcta ejecución, asegurando la compatibilidad con los recursos y dependencias necesarias.
  ![image](https://github.com/user-attachments/assets/6a651270-c768-46af-a8d6-44a4610cc546)
- En la siguiente carpeta tenemos almacenado el modelo de ia generativa y la base de chroma db, así como la base de datos que ocuparemos para entrenar el modelo.
  ![image](https://github.com/user-attachments/assets/a514925d-5aa2-49cf-9084-623a809413c8)
  ![image](https://github.com/user-attachments/assets/75fc8b46-0515-4aa3-891f-ec6cd7020136)
- Este es el directorio principal del proyecto, donde se tiene main.py el cual es el back end desarrollado en python
  ![image](https://github.com/user-attachments/assets/786518c6-9637-4422-8c6b-8318f0ed5a82)
- Dentro de la carpeta frontend tenemos toda la implementacion del front.
  ![image](https://github.com/user-attachments/assets/cf9c6020-ad4d-458a-995d-854abbacc86e)
- La aplicación desarrollada se encuentra principalmente en App.js y App.css.
- Una vez teniendo el desarrollo del back en main.py, desplegamos el servidor en el puerto 8000
  ![image](https://github.com/user-attachments/assets/8eaae925-faf1-4c8e-bf1f-2b0027032f1e)
- Se levanta el servidor:
  ![image](https://github.com/user-attachments/assets/ada6f7e5-9021-4585-aecd-10732fd8f80c)
- Posteriormente con ngrok exponemos el puerto para redes publicas.
  ![image](https://github.com/user-attachments/assets/fcbfa0c2-adc7-4db3-a69b-adf6a12aa069)
- Desplegamos el front end en el puerto 3000.
  ![image](https://github.com/user-attachments/assets/39fc8c8b-a523-4d09-aa23-34a22f7eb9f8)
- Y exponemos el puerto con ngrok nuevamente.
  ![image](https://github.com/user-attachments/assets/91cd0fcc-941e-4e13-8fdb-485b2b036662)
- Verificamos que el servidor del back end este funcionando correctamente.
  ![image](https://github.com/user-attachments/assets/d66439f6-c72a-489d-aaf8-6ebdee95175e)
- Verificamos que el servidor del front end este correcto.
  ![image](https://github.com/user-attachments/assets/37fbcbce-719c-4a99-8aec-44853f6d7998
  
 ##  Preguntas##
 
- Se hace la pregunta el servidor recibe la pregunta:
  ![image](https://github.com/user-attachments/assets/30fa7599-f491-46c0-99a5-39436ffc53d6)
- El servidor del back end recibe la pregunta:
  ![image](https://github.com/user-attachments/assets/aebb6156-7dcf-4f4b-bfd7-d0eb1f5fd10c)
- El servidor recibe la pregunta relacionada con la anterior.
  ![image](https://github.com/user-attachments/assets/b9285533-f665-4c81-8160-211e264639fc)
- El servidor recibe la pregunta y se implementa un sistema de memoria para que el back end guarde los ids de los casos que le dimos en la pregunta anterior.
  ![image](https://github.com/user-attachments/assets/533b0d77-fce3-4632-849b-c7c094d075f2)
- NOTA: Estas preguntas tienen una baja latencia porque ya se tienen mapeados lo casos en funciones dentro de python.
- Se envia la pregunta desde el back end.
  ![image](https://github.com/user-attachments/assets/d5a5e62b-c5be-4444-846f-d2f1e3a324f5)
- Se agrega una funcion para que el front end muestre mientras recibe la respuesta del back end, en estos casos siguientes la latencia es mayor ya que no tenemos mapeados al 100% estos casos.
  ![image](https://github.com/user-attachments/assets/75911199-d4fa-478d-a9c9-3273a0c6fb44)
- El servidor recibe la respuesta, aquí dejamos que el modelo haga su trabajo, es una pregunta que no conoce pero trata de responder.
  ![image](https://github.com/user-attachments/assets/ca998a9e-a0d6-40b9-b45c-2017d144e09b)
- Se envia la pregunta:
  ![image](https://github.com/user-attachments/assets/cdd1cfa5-5343-47a7-b022-0e1db187c0fc)
  ![image](https://github.com/user-attachments/assets/37cf594c-0ce0-45f3-940b-ba68013e05f3)
  ![image](https://github.com/user-attachments/assets/f09fdc05-3f8a-4311-945b-3c74633f44c8)
- El servidor recibe la pregunta sin conocerla antes:
  ![image](https://github.com/user-attachments/assets/10085417-2655-4e19-b607-890052758339)
  ![image](https://github.com/user-attachments/assets/a7ca1b17-25d5-4556-8e29-993891394778)
  ![image](https://github.com/user-attachments/assets/df45b8d6-0664-4ad8-9afb-40808519cc56)
Como podemos ver al aumentar la longitud de la pregunta disminuye el rendimiento del modelo de IA generativa, ademas de que no es capaz de contestar la pregunta en concreto.

# Informe

Dentro de un consultorio legal se busca optimizar la búsqueda de antecedentes y sentencias de los diversos casos que se les presenta, principalmente referente a temas de redes sociales. En la actualidad, los abogados hacen esta tarea de forma manual en una hoja de Excel para proporcionar detalles a sus clientes sobre las posibles demandas y penas que podrían estar enfrentando.

Teniendo en cuenta esto, la presente prueba plantea el uso de Inteligencia Artificial Generativa, de manera que el sistema pueda responder de forma automatizada y en un lenguaje coloquial, las preguntas más frecuentes sobre la historia de demandas y sus sentencias.

En esta prueba de concepto, se requiere que la herramienta responda a:
- Las sentencias de tres demandas específicas.
- El contexto de esas mismas tres demandas.
- Información detallada sobre un caso de acoso escolar (sentencia y detalles).
- La existencia (y detalle) de casos sobre el PIAR.

# Supuestos

1. **Disponibilidad y calidad de datos**  
   - Se toma por hecho que la base de datos con las demandas y sus sentencias está completa, correctamente etiquetada y organizada para extraer información relevante.

2. **Uso de un modelo de IA generativa**  
   - Se asume que el modelo seleccionado (BART-BASE-5) puede interpretar de forma adecuada las descripciones de los casos y responder con frases coherentes y contextualizadas.

3. **Lenguaje coloquial**  
   - Se asume que la herramienta debe reescribir la información en términos sencillos, sin tecnicismos, sin perder la precisión en lo legal, pero con un vocabulario que la mayoría de la gente pueda comprender.

4. **Limitaciones**  
   - Al ser una implementación inicial, se requiere un volumen de datos más robusto y variado para mejorar la precisión del modelo.

# Formas para resolver el caso y la opción tomada en esta prueba

Existen diferentes tipos de abordar problemas en cuanto a datos se refiere.

- **Uso de bases de datos relacionales para realizar búsquedas**
  - Consiste en realizar consultas a través de la base de datos usando palabras claves.
  - Con esta solución podemos hacer filtros, sin embargo, no ofrece explicaciones usando PLN (Procesamiento del lenguaje natural) ni la capacidad de resumir.

- **Sistemas de búsqueda semántica**
  - Utiliza un motor de búsqueda basado en la indexación de documentos por su contenido semántico.
  - Se puede localizar rápidamente la información relevante y puede integrarse con modelos generativos para responder de forma más natural.

- **Implementación de un chatbot con IA generativa**
  - Se integra un modelo generativo con una base de datos vectorial donde se pueden almacenar los datos.
  - El agente de IA busca la información en la base vectorial, posteriormente, el modelo de IA generativa construye la respuesta en lenguaje coloquial.

Para el caso de uso de esta prueba, se optó por la tercera opción.

En la gestión de embeddings y la recuperación eficiente de documentos, se utilizó **Chroma DB**, una base de datos vectorial diseñada para almacenamiento y consulta de datos semánticos en aplicaciones de IA generativa (Chroma Team, 2023). [3]

- Se creó un prototipo de chatbot que fuera capaz de:
  - Vectorizar los textos de las demandas y sentencias para facilitar la búsqueda semántica.
  - Realizar consultas a la base de datos vectorial con el fin de recuperar los documentos más relevantes.
  - Generar la respuesta con un modelo de lenguaje que simplifica y contextualiza la información. Según OpenAI (2023), la incorporación de modelos de gran tamaño (Large Language Models) mejora significativamente la generación de texto coherente. [4]
 
# Resultados del análisis de datos y de lo modelos
- **Calidad de búsqueda semántica:**  
  Al indexar el contenido de los casos en una base de datos vectorial, se incrementó la precisión en la identificación de demandas similares o relacionadas.

- **Respuestas en lenguaje natural:**  
  El modelo generativo demostró ser capaz de reformular las sentencias y resúmenes en un lenguaje muy natural, tal como se requería. La librería Transformers de Hugging Face facilita la integración de modelos de lenguaje de vanguardia en múltiples tareas de procesamiento de lenguaje natural (Hugging Face, n.d.). [5]

- **Desempeño:**  
  Para un conjunto de datos de tamaño modero, la latencia con la que responde fue aceptable. Sin embargo, en situaciones con más volumen de datos, sería necesario optimizar tanto la indexación como la generación de texto para mantener la eficacia de las respuestas.

## Resultado Análisis de datos ##

Se realizo un análisis de datos el cual esta dentro del repositorio llamado "Analisis_Sentencias.ipynb", los resultados fueron los siguientes: 

  ### Análisis de las sentencias recibidas por año

La gráfica refleja un crecimiento en el número de sentencias desde 2019, con un aumento sostenido hasta alcanzar su punto más alto en 2023, superando las 80 resoluciones. En 2024 se observa una ligera disminución, aunque el número de sentencias sigue siendo significativamente mayor en comparación con los años previos a 2022. A pesar de la caída en 2024, el nivel de sentencias sigue siendo elevado, lo que podría indicar una estabilización en la capacidad del sistema interno de los abogados para gestionar demandas.

![image](https://github.com/user-attachments/assets/831d08b3-4e0f-4061-a58b-996a658e8729)

# Análisis de sentencias por mes

La gráfica muestra un aumento en las sentencias gestionadas por el despacho desde 2019, con un pico en 2023 y estabilización en 2024. Los meses con más actividad son agosto, septiembre y octubre, mientras que enero y febrero presentan menos casos. El crecimiento desde 2022 sugiere una mayor demanda de servicios o mejoras en los procesos internos.

![image](https://github.com/user-attachments/assets/90ffed15-7f09-4e88-91b6-ed9e7e1daf60)

# Resultado del anális de los temas mas comunes

La mayoría de los casos gestionados por el despacho están relacionados con la competencia de la jurisdicción contencioso-administrativa y acciones de tutela. Sin embargo, la gran proporción de sentencias clasificadas como desconocido lo que que podría ser útil mejorar la categorización de los casos para obtener un análisis más preciso de las tendencias legales.

![image](https://github.com/user-attachments/assets/245cf2b7-23e4-4495-a6de-8436bfb98a8b)

# Análisis de la Nube de Palabras de los temas y sintesis de los casos

La nube de palabras muestra que los casos del despacho se enfocan en derechos fundamentales, regulación estatal y conflictos digitales, destacando temas como emergencia económica, libertad de expresión y redes sociales.

![image](https://github.com/user-attachments/assets/ce116bde-6ee0-4442-af26-12e9dec656bd)

![image](https://github.com/user-attachments/assets/677e8325-398c-43cd-be7a-ce04a1803b2c)

# Resultados del Análisis de los Modelos

## Evaluación del Rendimiento del Sistema

Durante la prueba técnica, se analizaron distintos aspectos del sistema, desde la capacidad de procesamiento hasta la latencia en la obtención de respuestas por parte del modelo de IA generativa. A continuación, se presentan los principales hallazgos:

- **Desempeño de la Máquina Virtual:**  
  La configuración utilizada en Azure (Standard D4as v6 con 4 vCPU y 16 GiB de RAM) permitió un rendimiento estable del modelo de IA generativa. Sin embargo, cuando se realizaron consultas más extensas, se observó un incremento en la latencia de respuesta.

- **Almacenamiento y Disponibilidad de Datos:**  
  La integración de ChromaDB para la indexación y recuperación de sentencias almacenadas permitió optimizar el acceso a la información. No obstante, en algunos casos específicos, el sistema no encontró coincidencias exactas, lo que afectó la calidad de las respuestas.

## Análisis de la Precisión y Calidad de las Respuestas

### **Consultas con IDs específicos de casos legales**  
- Cuando se preguntó por sentencias específicas (ej. "¿Cuál fue la sentencia de las siguientes demandas: T-185/22, T-351/21, T-351/22?"), el sistema fue capaz de extraer correctamente la información almacenada en ChromaDB.  
- Sin embargo, en casos donde no existía una coincidencia exacta, el modelo generaba respuestas genéricas, indicando que no encontró información relevante.

### **Consultas de seguimiento**  
- Se implementó un sistema de memoria que permite recordar los IDs consultados en la pregunta anterior.  
- Gracias a esto, cuando se le preguntó "¿De qué se trataron las 3 demandas anteriores?", el sistema fue capaz de recuperar los detalles correspondientes a las sentencias previamente mencionadas.

### **Consultas de contexto general**  
- Preguntas más amplias, como "¿Diga el detalle de la demanda relacionada con acoso escolar?", fueron procesadas de manera efectiva cuando la información estaba disponible en la base de datos.  
- En algunos casos, el modelo generó respuestas excesivamente detalladas, por lo que se ajustó la configuración para priorizar respuestas más concisas y naturales.

## Impacto del Tamaño de la Pregunta en el Rendimiento

Se identificó que la latencia del sistema aumentaba cuando las preguntas contenían múltiples IDs o eran demasiado extensas. Esto se debe a:

- **Mayor cantidad de consultas a ChromaDB:** Cuando se requiere extraer información sobre múltiples sentencias en una sola consulta, el sistema necesita realizar varias búsquedas simultáneas.
- **Procesamiento del modelo de IA generativa:** Preguntas más largas implican un procesamiento más complejo en el modelo, lo que puede generar respuestas menos precisas o incoherentes.

Para mitigar este problema, se implementaron las siguientes soluciones:

- **Límite de longitud en las respuestas:** Se ajustó el modelo para proporcionar respuestas más cortas y directas, evitando la sobrecarga de información innecesaria.
- **Preprocesamiento de preguntas:** Se mejoró la extracción de IDs de sentencias para optimizar las búsquedas en ChromaDB.
- **Mensajes en el frontend:** Se agregó un indicador de Escribiendo... mientras el backend procesa la consulta, mejorando la experiencia del usuario.


Los resultados obtenidos demuestran que la integración de ChromaDB y un modelo de IA generativa es una solución viable para responder preguntas sobre sentencias legales de manera eficiente. Sin embargo, se identificaron áreas de mejora, como:

- **Optimización del tiempo de respuesta** para consultas extensas.
- **Mejora en la coherencia de las respuestas** cuando no hay coincidencias exactas en la base de datos.
- **Mayor control sobre la longitud y claridad** de las respuestas generadas

# Futuros Ajustes o Mejoras

- **Ampliar la base de datos**  
  Incluir más casos de demandas para que el modelo pueda aprender a dar respuestas más contextualizadas.

- **Filtrar información**  
  Añadir un mecanismo para que datos sensibles de los clientes puedan ser anónimos antes de presentar la información.

- **Entrenamiento**  
  Ajustar el modelo con más ejemplos donde se use lenguaje legal y sus equivalentes en lenguaje coloquial para mejorar la precisión de las respuestas.

- **Sistema**  
  Implementar un sistema de retroalimentación que permita a los abogados calificar o validar las respuestas generadas con la finalidad de mejorar la exactitud del modelo.

---

## Apreciaciones y Comentarios del Caso  

La implementación de IA generativa reduce el tiempo de búsqueda y proporciona una explicación más amigable para el usuario final. Además, este enfoque presenta escalabilidad y rentabilidad, pues, al entrenar el modelo con más datos, tiene el potencial de convertirse en una herramienta que facilite las tareas diarias tanto de los abogados como de los usuarios. Para finalizar, esta herramienta tiene la capacidad de expandirse, ya que, si bien su uso actual es para temas legales, la metodología basada en búsqueda semántica y modelos de IA generativa resulta aplicable a numerosos campos de estudio.

---

## Conclusión  

En este informe se describe la solución propuesta para automatizar las consultas del historial de demandas y sentencias mediante el uso de IA generativa. La prueba de concepto demuestra la viabilidad de aplicar técnicas de búsqueda semántica y modelos generativos, en especial para agilizar trabajos repetitivos como consultas y proporcionar respuestas en lenguaje coloquial. A medida que se haga más robusta la base de datos y se realicen diversos ajustes en el modelo, el resultado podría ser una mayor precisión, haciendo que la herramienta sea aún más útil.

---

## Referencias  

[1] Hugging Face AI. (s.f.). mBART-50 one to many multilingual machine translation. https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt

[2] IBM. (s.f.). Chatbot Limitations: Challenges and Considerations. https://www.ibm.com/mx-es/topics/chatbots

[3] Chroma Team. (2023). *Chroma Documentation*. [https://docs.trychroma.com](https://docs.trychroma.com)  

[4] OpenAI. (2023). *ChatGPT: Introducing a new language model*. OpenAI. [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt)  

[5] Hugging Face. (n.d.). *Transformers: State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX*. [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)  









