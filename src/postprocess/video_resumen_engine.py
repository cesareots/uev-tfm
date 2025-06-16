import logging
from pathlib import Path

from moviepy import VideoClip, VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import *

from src.utils import utils as ut
from src.utils.constants import SOCCERNET_LABELS, SOCCERNET_LABELS_ES

logger = logging.getLogger(__name__)


def fn_multiply_speed(
        clip: VideoClip,
        buffer_seconds: float = 2.0,
        factor_velocidad: float = 0.25,
) -> CompositeVideoClip:
    # Separar el clip del evento en 3 partes:
    # Parte 1: antes del momento clave (velocidad normal)
    parte1 = clip.subclipped(0, buffer_seconds)

    # Parte 2: momento clave (aplicamos slow motion)
    parte_slowmo = clip.subclipped(buffer_seconds, clip.duration - buffer_seconds)
    # parte_slowmo_procesada = parte_slowmo.fx(speedx, factor_velocidad)
    parte_slowmo_procesada = parte_slowmo.with_effects([
        MultiplySpeed(factor=factor_velocidad)
    ])

    # Parte 3: después del momento clave (velocidad normal)
    parte3 = clip.subclipped(clip.duration - buffer_seconds)

    # Unir las tres partes
    clip_final = concatenate_videoclips([parte1, parte_slowmo_procesada, parte3])
    # print(f"parte1: {parte1.fps}")
    # print(f"parte_slowmo_procesada: {parte_slowmo_procesada.fps}")
    # print(f"parte3: {parte3.fps}")
    # print(f"clip_final: {clip_final.fps}")

    # NOTA SOBRE EL AUDIO: El slow motion alarga el vídeo pero no el audio.
    # Esta línea re-asigna el audio original al nuevo clip. El audio se desincronizará durante la parte lenta y se cortará si el vídeo final es más largo.
    # Para un resultado profesional, a menudo se añade música de fondo en su lugar.
    #if clip.audio:
        #clip_final = clip_final.with_audio(clip.audio)

    return clip_final


def fn_slide_in(
        clip: VideoClip,
        duration: float,
        direction: str,
        screen_size: tuple[int, int]
) -> CompositeVideoClip:
    """
    Anima un clip para que se deslice hacia el centro de la pantalla.
    """
    screen_width, screen_height = screen_size
    clip_width, clip_height = clip.size

    # Definir la posición inicial y final
    start_pos_x = {'left': -clip_width, 'right': screen_width, 'top': 'center', 'bottom': 'center'}[direction]
    start_pos_y = {'left': 'center', 'right': 'center', 'top': -clip_height, 'bottom': screen_height}[direction]
    end_pos_x = 'center'
    end_pos_y = 'center'

    # Función de posición que cambia con el tiempo 't'
    def position_func(t):
        if t < duration:
            # Interpolar posición X
            current_x = 'center'
            if isinstance(start_pos_x, int):
                # Calcular la progresión lineal
                progress = t / duration
                current_x = int(start_pos_x + (clip.w / 2 + screen_width / 2 - start_pos_x) * progress)

            # Interpolar posición Y
            current_y = 'center'
            if isinstance(start_pos_y, int):
                progress = t / duration
                current_y = int(start_pos_y + (clip.h / 2 + screen_height / 2 - start_pos_y) * progress)

            return (current_x, current_y)
        else:
            return (end_pos_x, end_pos_y)

    animated_clip = clip.with_position(position_func)

    # El clip animado debe estar sobre un fondo del tamaño de la pantalla
    return CompositeVideoClip([animated_clip], size=screen_size).with_duration(clip.duration)


def video_resumen_moviepy(
        final_events,
        original_video_path: Path,
        buffer_seconds: float = 2.0,
        transition_seconds: float = 0.5,
):
    output_path = original_video_path.parent / f"{original_video_path.stem}_resumen_final.mp4"
    eventos_clave = ut.extraer_claves_ordenadas(SOCCERNET_LABELS)

    # Cargar el vídeo original una sola vez
    try:
        main_video_clip = VideoFileClip(str(original_video_path))
    except Exception as e:
        logger.error(f"Al cargar el vídeo original '{original_video_path}': {e}")
        return

    final_clips = []
    video_duration = main_video_clip.duration
    screen_size = main_video_clip.size

    # Procesar cada evento para crear subclips
    try:
        for i, event in enumerate(final_events):
            evento_str = event.get("evento", "Evento Desconocido")
            inicio = event.get("inicio", 0)
            fin = event.get("fin", 0)

            # Añadir buffer (margen de tiempo) asegurando no salirse de los límites del vídeo
            start_with_buffer = max(0, inicio - buffer_seconds)
            end_with_buffer = min(video_duration, fin + buffer_seconds)

            if start_with_buffer >= end_with_buffer:
                logger.warning(
                    f"Saltando evento '{evento_str}' en {inicio:.2f}s porque su duración con buffer es inválida.")
                continue

            logger.info(
                f"Procesando evento {i + 1}/{len(final_events)}: '{evento_str}' de {start_with_buffer:.2f}s a {end_with_buffer:.2f}s")

            subclip = main_video_clip.subclipped(start_with_buffer, end_with_buffer)

            txt_clip = TextClip(
                # font='Arial-Bold',
                # text=f"{evento_str}",
                text=f"{SOCCERNET_LABELS_ES.get(evento_str)}",
                font_size=90,
                # color="yellow",
                color="#FFD700",
                # bg_color="#ACFAC8",
                stroke_color="black",
                stroke_width=2,
                # margin=("left", "top", "right", "bottom"),
                # size=(25,25),
                # method="caption",
                # horizontal_align="left",
                # horizontal_align="center",
                # horizontal_align="right",
                # vertical_align="top",
                # vertical_align="center",
                # vertical_align="bottom",
                duration=subclip.duration,
            )
            """.with_effects([
                FadeIn(duration=transition_seconds),
                #CrossFadeIn(duration=transition_seconds),
                FadeOut(duration=transition_seconds),
            ])"""

            """# en caso que no ingrese a ningun 'case' (es raro)
            temporal = CompositeVideoClip([subclip, txt_clip])
            temporal = temporal.with_effects([
                FadeIn(duration=transition_seconds),
                FadeOut(duration=transition_seconds),
            ])
            list_effects=[temporal]"""

            effect=None
            list_effects = []

            match evento_str:
                case "Goal":
                    # video.fx.MultiplySpeed: multiplica la velocidad del clip por un factor
                    effect = fn_multiply_speed(
                        clip=subclip,
                        buffer_seconds=buffer_seconds,
                        factor_velocidad=0.25,
                    )
                    list_effects.append(effect)

                    # video.fx.FreezeRegion: Congela una parte específica de la pantalla mientras el resto del vídeo continúa reproduciéndose.

                    # video.fx.TimeMirror: Reproduce el clip normalmente y luego lo reproduce en reversa, creando un efecto de espejo en el tiempo.

                    # video.fx.Blink: Hace que el clip "parpadee", alternando rápidamente entre sus fotogramas y un fotograma negro.

                case "Yellow card":
                    # video.fx.MultiplySpeed
                    effect = fn_multiply_speed(
                        clip=subclip,
                        buffer_seconds=buffer_seconds,
                        factor_velocidad=0.5,
                    )
                    list_effects.append(effect)

                    # video.fx.Blink

                case "Substitution":
                    # video.fx.SlideIn: Hace que el clip entre en la pantalla deslizándose desde un lado.
                    effect = fn_slide_in(
                        clip=subclip,
                        duration=2.0,
                        direction="top",  # TODO que sea random
                        screen_size=screen_size,
                    )
                    list_effects.append(effect)

                    # video.fx.SlideOut: Hace que el clip salga de la pantalla deslizándose hacia un lado.

                    pass
                case "Corner":
                    # video.fx.FreezeRegion
                    pass

                case _:
                    pass

            list_effects.append(txt_clip)

            effects_with_text = CompositeVideoClip(list_effects).with_effects([
                FadeIn(duration=transition_seconds),
                FadeOut(duration=transition_seconds),
            ])

            final_clips.append(effects_with_text)
    except Exception as e:
        logger.error(e)

    if not final_clips:
        logger.error("No se pudo crear ningún subclip válido. Abortando la creación del vídeo.")
        main_video_clip.close()
        return

    try:
        # Concatenar todos los subclips con una transición de fundido cruzado
        # print(final_clips)
        logger.info("Concatenando todos los subclips en el video resumen final.")
        # summary_video = concatenate_videoclips(final_clips, transition=fadein(transition_seconds), method="compose")
        summary_video = concatenate_videoclips(
            clips=final_clips,
            # method="chain",
            method="compose",
            # transition=CrossFadeIn(duration=transition_seconds),  # TODO corregir o desestimarlo
            # transition=txt_clip,
            # padding=-transition_seconds,
        )
    except Exception as e:
        logger.error(e)

    try:
        # Codecs estándar para alta compatibilidad (vídeo H.264, audio AAC)
        summary_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
        )
        logger.info("¡Video-resumen generado exitosamente!")
    except Exception as e:
        logger.error(e)
    finally:
        # Liberar recursos
        main_video_clip.close()
        summary_video.close()
        for clip in final_clips:
            clip.close()
