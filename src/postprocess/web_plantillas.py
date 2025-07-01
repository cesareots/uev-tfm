# Los marcadores de posición como {video_summary_path} se reemplazarán con Python.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen de Partido de Fútbol</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f0f2f5;
            color: #1c1e21;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #0d2c54;
            border-bottom: 2px solid #e7f3ff;
            padding-bottom: 10px;
        }}
        video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .interactive-player {{
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }}
        .video-container {{
            flex: 3; /* Ocupa 3/4 del espacio */
        }}
        .event-list {{
            flex: 1; /* Ocupa 1/4 del espacio */
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            max-height: 500px; /* Altura máxima para la lista de eventos */
            overflow-y: auto; /* Scroll si hay muchos eventos */
        }}
        .event-list ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .event-list li a {{
            display: block;
            padding: 10px;
            cursor: pointer;
            text-decoration: none;
            color: #053b50;
            border-bottom: 1px solid #ddd;
            transition: background-color 0.2s;
        }}
        .event-list li a:hover {{
            background-color: #e9ecef;
        }}
        .event-time {{
            font-size: 0.8em;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Detector de Eventos Clave en Fútbol</h1>
        
        <h2>Vídeo Resumen Generado</h2>
        <video controls src="{video_summary_path}" poster="">
            Tu navegador no soporta el tag de video.
        </video>

        <h2>Reproductor Interactivo del Partido Completo</h2>
        <div class="interactive-player">
            <div class="video-container">
                <video id="fullVideoPlayer" controls src="{full_video_path}" poster="">
                    Tu navegador no soporta el tag de video.
                </video>
            </div>
            <div class="event-list">
                <h3>Eventos Detectados</h3>
                <ul>
                    {event_list_html}
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Este simple script de JavaScript controla el reproductor interactivo
        function jumpToTime(seconds) {{
            const videoPlayer = document.getElementById('fullVideoPlayer');
            videoPlayer.currentTime = seconds;
            videoPlayer.play();
        }}
    </script>
</body>
</html>
"""

HTML_TEMPLATE_2 = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen de Partido - TFM</title>
    <style>
        :root {{
            --dark-bg: #1a1a1a;
            --primary-card-bg: #242424;
            --secondary-card-bg: #333333;
            --text-primary: #e0e0e0;
            --text-secondary: #b3b3b3;
            --accent-color: #00aaff;
            --border-color: #444444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--dark-bg);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: auto;
            background: var(--primary-card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 2.5em;
            color: #ffffff;
            margin: 0;
        }}
        h2 {{
            font-size: 1.8em;
            color: var(--accent-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
            background: #000;
        }}
        .interactive-player {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: flex-start;
        }}
        .video-container {{
            flex: 3;
            min-width: 300px; /* Para mejor responsiveness */
        }}
        .event-list {{
            flex: 1;
            background-color: var(--secondary-card-bg);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            min-width: 250px;
            max-height: 600px;
            overflow-y: auto;
        }}
        .event-list h3 {{
            margin-top: 0;
            color: #f5f5f5;
            text-align: center;
        }}
        .event-list ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .event-item a {{
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px;
            cursor: pointer;
            text-decoration: none;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.2s ease-in-out;
        }}
        .event-item a:hover {{
            background-color: rgba(0, 170, 255, 0.1);
        }}
        .event-icon {{
            font-size: 1.5em;
        }}
        .event-details {{
            display: flex;
            flex-direction: column;
        }}
        .event-name {{
            font-weight: 600;
        }}
        .event-time {{
            font-size: 0.85em;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Análisis de Partido de Fútbol</h1>
            <p>Resultado del Entrenamiento del Modelo de Detección de Eventos</p>
        </div>
        
        <h2>Reproductor Interactivo</h2>
        <div class="interactive-player">
            <div class="video-container">
                <video id="fullVideoPlayer" controls src="{full_video_path}">
                    Tu navegador no soporta el tag de video.
                </video>
            </div>
            <div class="event-list">
                <h3>Eventos Detectados</h3>
                <ul>
                    {event_list_html}
                </ul>
            </div>
        </div>

        <h2>Vídeo Resumen Generado</h2>
        <video controls src="{video_summary_path}">
            Tu navegador no soporta el tag de video.
        </video>
    </div>

    <script>
        function jumpToTime(event, seconds) {{
            event.preventDefault();
            const videoPlayer = document.getElementById('fullVideoPlayer');
            videoPlayer.currentTime = seconds;
            videoPlayer.play();
        }}
    </script>
</body>
</html>
"""

HTML_TEMPLATE_3 = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen de Partido - TFM</title>
    <style>
        :root {{
            --dark-bg: #1a1a1a;
            --primary-card-bg: #242424;
            --secondary-card-bg: #333333;
            --text-primary: #e0e0e0;
            --text-secondary: #b3b3b3;
            --accent-color: #00aaff;
            --border-color: #444444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--dark-bg);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: auto;
            background: var(--primary-card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 2.5em;
            color: #ffffff;
            margin: 0;
        }}
        h2 {{
            font-size: 1.8em;
            color: var(--accent-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        video {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
            background: #000;
        }}
        .interactive-player {{
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: flex-start;
        }}
        .video-container {{
            flex: 3;
            min-width: 300px; /* Para mejor responsiveness */
        }}
        .event-list {{
            flex: 1;
            background-color: var(--secondary-card-bg);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            min-width: 250px;
            max-height: 600px;
            overflow-y: auto;
        }}
        .event-list h3 {{
            margin-top: 0;
            color: #f5f5f5;
            text-align: center;
        }}
        .event-list ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .event-item a {{
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px;
            cursor: pointer;
            text-decoration: none;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.2s ease-in-out;
        }}
        .event-item a:hover {{
            background-color: rgba(0, 170, 255, 0.1);
        }}
        .event-icon {{
            font-size: 1.5em;
        }}
        .event-details {{
            display: flex;
            flex-direction: column;
        }}
        .event-name {{
            font-weight: 600;
        }}
        .event-time {{
            font-size: 0.85em;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Análisis de Partido de Fútbol</h1>
            <p>Resultado del Entrenamiento del Modelo de Detección de Eventos</p>
        </div>
        
        <h2>Vídeo Resumen Generado</h2>
        <video controls src="{video_summary_path}">
            Tu navegador no soporta el tag de video.
        </video>

        <h2>Reproductor Interactivo</h2>
        <div class="interactive-player">
            <div class="video-container">
                <video id="fullVideoPlayer" controls src="{full_video_path}">
                    Tu navegador no soporta el tag de video.
                </video>
            </div>
            <div class="event-list">
                <h3>Eventos Detectados</h3>
                <ul>
                    {event_list_html}
                </ul>
            </div>
        </div>
    </div>

    <script>
        function jumpToTime(event, seconds) {{
            event.preventDefault();
            const videoPlayer = document.getElementById('fullVideoPlayer');
            videoPlayer.currentTime = seconds;
            videoPlayer.play();
        }}
    </script>
</body>
</html>
"""
