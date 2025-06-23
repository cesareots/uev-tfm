# uev - tfm - detector y video resumen de eventos clave en partidos de futbol ⚽🎥🤖
- Computer Vision & Artificial Intelligence.
- Este proyecto da vida a la inteligencia artificial para revolucionar la forma en que consumimos el fútbol. Partiendo de horas de video de partidos en crudo del dataset SoccerNet, el sistema automatiza un completo flujo de trabajo: desde la descarga y el procesamiento inteligente de datos para crear un dataset de acciones clave, hasta el entrenamiento de avanzados modelos de Deep Learning (CNN 3D y R(2+1)D ResNet) capaces de "entender" el juego.
- El resultado final es un motor autónomo que analiza cualquier nuevo partido, detecta sus momentos culminantes y genera automáticamente un resumen de video dinámico y editado profesionalmente, con efectos visuales y anotaciones, transformando datos brutos en una experiencia de visualización cautivadora.

### variables de entorno
- Se maneja un único .env para todo el proyecto. Estas variables son leídas mediante el archivo 'constants.py' y luego utilizadas en cualquier punto.

### logger
- Cada script 'main...' configura su propio logger. Luego cada main realiza verificaciones antes de iniciar el sistema.

### constants.py
- Centraliza todas las variables de configuración del proyecto, como rutas de directorios, parámetros de modelos, umbrales de confianza y etiquetas de clases. Sirve como la única fuente de verdad para las constantes usadas a lo largo de toda la aplicación.

### utils.py
- Provee una colección de funciones de utilidad reutilizables en todo el proyecto. Se encarga de tareas comunes como la gestión de archivos y directorios, la escritura en formatos CSV/JSON/TXT y la validación de argumentos, actuando como la caja de herramientas general del sistema.

### main_soccernet_generar_clip.py
- Es el punto de partida del flujo de datos. Descarga videos de partidos de SoccerNet y, basándose en sus archivos de anotaciones, recorta y guarda automáticamente clips cortos de acciones clave (goles, tarjetas, etc.), creando el dataset base para el entrenamiento de los modelos.

### main_soccernet_eliminar_video.py
- Es un script de limpieza para gestionar el espacio en disco. Lee una lista de partidos y, previa confirmación, borra de forma permanente los pesados archivos de video originales que ya no son necesarios una vez que los clips de acciones han sido generados.

### main_soccernet_recortar_video.py
- Permite crear fragmentos de video con una duración y punto de inicio personalizados a partir de los videos completos de SoccerNet. Su función es generar muestras de video específicas, principalmente para realizar pruebas de inferencia rápidas y controladas.

### dataset_soccernet.py
- Contiene la clase Dataset de PyTorch, que actúa como puente entre los videoclips guardados en disco y los modelos. Se encarga de cargar los clips, aplicarles las transformaciones adecuadas y dividirlos de forma estratificada en conjuntos de entrenamiento, validación y prueba.

### transforms.py
- Define las distintas cadenas de transformaciones y aumento de datos para los videos. Proporciona pipelines de preprocesamiento específicas para cada arquitectura de modelo (CNN3D y ResNet), asegurando que los datos de entrada se normalicen y aumenten adecuadamente antes del entrenamiento.

### engine_training.py
- Es el motor de entrenamiento genérico del proyecto. Contiene la lógica para ejecutar los bucles de entrenamiento y validación, calcular métricas, ajustar la tasa de aprendizaje y gestionar el guardado de checkpoints del modelo, siendo invocado por los scripts de cada arquitectura.

### main_cnn3d.py
- Define una arquitectura de Red Neuronal Convolucional 3D personalizada. Orquesta todo el proceso de entrenamiento para este modelo específico: configura las transformaciones, prepara los dataloaders y utiliza el engine_training para entrenar el modelo desde cero.

### main_resnet.py
- Implementa la estrategia de Transfer Learning usando un modelo R(2+1)D ResNet pre-entrenado. Configura el fine-tuning de capas, establece optimizadores con tasas de aprendizaje diferenciales y llama al engine_training para adaptar el modelo a la tarea de clasificación de acciones.

### main_transforms_visualize.py
- Es un script de depuración para visualizar el efecto de las transformaciones de datos. Carga muestras de video, les aplica las cadenas de aumento definidas en transforms.py y guarda imágenes de los fotogramas resultantes para verificar que el preprocesamiento funciona correctamente.

### main_infe.py
- Es el script de inferencia que utiliza un modelo ya entrenado. Carga un checkpoint, analiza un video completo mediante una ventana deslizante para detectar eventos y post-procesa las predicciones para obtener una lista final de acciones significativas con sus marcas de tiempo.

### video_resumen_engine.py
- Representa la etapa final del proyecto. Recibe la lista de eventos detectados por el script de inferencia y, usando MoviePy, edita el video original para crear un resumen automático de los momentos más destacados, aplicando efectos visuales, texto y animaciones.
