# uev-tfm
- Detector de eventos clave en partidos de futbol
- Computer Vision
- Artificial Intelligence
- Tuve que bajar a 95% la capacidad máxima del CPU (ThinkPad P16s Gen3) para que no salga pantalla azul y se reinicie el sistema operativo, este cambio lo hize desde 'control panel/power options/edit plan settings/change advanced power settings'

### variables de entorno
- Se maneja un único .env para todo el proyecto. Estas variables son leídas mediante el archivo 'constants.py' y luego utilizadas en cualquier punto.
- 

### utils.py
- Utilizado desde cualquir punto del sistema, contiene principalmente funciones reutilizables para modular la arquitectura del proyecto.
- 

### logger
- Cada script 'main...' configura su propio logger.
- Luego cada main realiza verificaciones antes de iniciar el sistema.
- 

### main_soccernet.py
- Descarga archivos por cada partido: videos, json, etc.
- Se pueden descargar los primeros N partidos, o también indicando los índices de cada partido deseado.
- Se establece la lista de acciones a utilizar; soccernet tiene 17 acciones etiquetadas en cada archivo json, por cada partido.
- Se recorren los partidos descargados, y por cada acción se generan clips de +-2 segundos, que luego servirán para entrenamiento del modelo-IA.
- 

### main_detectando.py
- Es solo para demostrar el funcionamiento de YOLO; carga un modelo YOLO, carga un video X, se establece un tiempo, recorre cada frame anotando las detecciones.
- 

### main_yolo_export.py
- si se llega a utilizar YOLO, servirá para exportar el modelo final al tipo OpenVINO.
- En este tipo de modelo se puede utilizar la NPU de mi ThinkPad P16s Gen 3.
- 
