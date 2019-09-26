# FAK-NES
## Predictor de Puntaje de Rendimiento de Noticias en Facebook

Se utilizaron más de 900,000 datos de publicaciones de noticias de Periódicos Mexicanos

# Descripción

Análisis estadístico de un DataSet obtenido con la herramienta CrowdTangle para llegar a un modelo predictivo del puntaje de la publicación
otorgada por Facebook tomando en cuenta diferentes variables como el tipo de video, descripción, y la variable más importante
un puntaje por medio de Análisis de Sentimiento en el que se clasificaban las noticias o publicaciones según el título. Las clasificaciones
de títulos se encontraba entre una de las siguientes: Positivo, Neutro o Negativo.

# Metodología

| Pasos | Descripción |
| --- | --- |
| <b>Adquisición de Datos<b> | Obtención de datos de aproximadamente un año de publicaciones de 129 periódicos Mexicanos utilizando la herramienta CrowdTangle |
| Exploración de Datos | Exploración de los datos para entender cómo está conformado el DataSet |
| Análisis de Datos | Una vez terminado el proceso de análisis y limpieza de datos se procede a analizar con herramientas estadísticas todos los datos para prepararlos para el modelo |
| Modelado de Datos | Creación de Modelo de Machine Learning de Regresión usando Random Tree Classifier para la predicción del puntaje |
