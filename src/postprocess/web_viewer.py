import argparse
import json
import logging
from pathlib import Path

from src.utils import utils as ut
from src.utils.constants import LOG_DIR, LOG_VISOR_WEB
from src.postprocess.web_plantillas import HTML_TEMPLATE, HTML_TEMPLATE_2, HTML_TEMPLATE_3

logger = logging.getLogger(__name__)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_VISOR_WEB,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
    )


def generate_web_viewer(
        events_path: Path,
        video_summary_path: Path,
        full_video_path: Path,
        output_html_path: Path,
):
    """
    Genera un archivo HTML interactivo para visualizar los resultados.
    """
    logger.info(f"Iniciando generación de visor web en '{output_html_path}'")

    # 1. Cargar la lista de eventos desde el archivo JSON
    try:
        with open(events_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo de eventos en '{events_path}'")
        return

    # 2. Generar el HTML para la lista de eventos
    event_list_html = ""
    
    event_icons = {
        "Goal": "⚽️",
        "Yellow card": "🟨",
        "Substitution": "🔄",
        "Corner": "🚩",
        "Default": "▶️",
    }
    
    if not events:
        event_list_html = "<li>No se detectaron eventos significativos.</li>"
    else:
        for event in events:
            evento_str = event.get('evento', 'Desconocido')
            inicio_s = event.get('inicio', 0)
            
            # Formatear el tiempo a MM:SS
            minutos = int(inicio_s // 60)
            segundos = int(inicio_s % 60)
            timestamp_str = f"{minutos:02d}:{segundos:02d}"

            # Obtener el icono correspondiente o uno por defecto
            icon = event_icons.get(evento_str, event_icons["Default"])

            event_list_html += f'''
            <li class="event-item">
                <a href="#" onclick="jumpToTime(event, {inicio_s})"> 
                    <span class="event-icon">{icon}</span>
                    <div class="event-details">
                        <span class="event-name">{evento_str}</span>
                        <span class="event-time">{timestamp_str}</span>
                    </div>
                </a>
            </li>
            '''

    # 3. Rellenar la plantilla HTML principal con las rutas y la lista de eventos
    # Usar rutas relativas es mejor si el HTML y los vídeos estarán en la misma carpeta
    final_html = HTML_TEMPLATE_3.format(
        video_summary_path=Path(video_summary_path).name,
        full_video_path=Path(full_video_path).name,
        event_list_html=event_list_html,
    )

    # 4. Escribir el archivo HTML final
    try:
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        logger.info(f"Visor web generado exitosamente en: '{output_html_path}'")
        logger.info(
            "Abre ese archivo en tu navegador. Asegúrate de que los vídeos estén en la misma carpeta que el HTML.")
    except Exception as e:
        logger.error(f"Error al escribir el archivo HTML: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genera un visor web interactivo para el resumen de un partido.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events_json",
        default="inference/spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna/11 minutos/2_720p_recortado_predictions_post_process.json",
        type=str,
        help="Ruta al archivo JSON con los eventos detectados.",
    )
    parser.add_argument(
        "--summary_video",
        default="inference/spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna/11 minutos/2_720p_recortado_resumen_final.mp4",
        type=str,
        help="Ruta al video resumen generado.",
    )
    parser.add_argument(
        "--full_video",
        default="inference/spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna/11 minutos/2_720p_recortado_optimizado.mp4",
        type=str,
        help="Ruta al video original del partido.",
    )
    parser.add_argument(
        "--output_html",
        default="inference/spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna/11 minutos/resumen_interactivo.html",
        type=str,
        help="Nombre del archivo HTML de salida.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()

    generate_web_viewer(
        events_path=Path(args.events_json),
        video_summary_path=Path(args.summary_video),
        full_video_path=Path(args.full_video),
        output_html_path=Path(args.output_html),
    )
