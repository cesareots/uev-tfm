import logging
from pathlib import Path

from moviepy import VideoClip, VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
from moviepy.video.fx import *

from src.utils import utils as ut
from src.utils.constants import SOCCERNET_LABELS_ES, PATH_RESOURCES

logger = logging.getLogger(__name__)


def fn_multiply_speed_blink(
        clip: VideoClip,
        buffer: float = 1.5,
        factor_velocidad: float = 0.5,
        duration_blink: float = 0.2,
) -> CompositeVideoClip:
    # video.fx.MultiplySpeed: multiplica la velocidad del clip por un factor
    # video.fx.Blink: Hace que el clip "parpadee", alternando rápidamente entre sus fotogramas y un fotograma negro.

    # intuyendo que el evento clave ocurre en la mitad de la duracion del video
    clave_ini = clip.duration / 2 - buffer
    clave_fin = clip.duration / 2 + buffer

    buffer_2 = buffer / 2
    clave_ini_2 = clip.duration / 2 - buffer_2
    clave_fin_2 = clip.duration / 2 + buffer_2

    # parte 1: antes del momento clave (velocidad normal)
    parte1 = clip.subclipped(0, clave_ini)

    # parte 2: momento clave (aplicamos slow motion)
    parte_slowmo_a = clip.subclipped(clave_ini, clave_ini_2)
    parte_slowmo_a_procesada = parte_slowmo_a.with_effects([
        MultiplySpeed(factor=factor_velocidad),
        Blink(duration_on=duration_blink, duration_off=duration_blink)
    ])

    # parte 3: momento clave (aplicamos slow motion)
    parte_slowmo_b = clip.subclipped(clave_ini_2, clave_fin_2)
    parte_slowmo_b_procesada = parte_slowmo_b.with_effects([
        MultiplySpeed(factor=factor_velocidad / 2),
        Blink(duration_on=duration_blink / 2, duration_off=duration_blink / 2)
    ])

    # parte 4: momento clave (aplicamos slow motion)
    parte_slowmo_c = clip.subclipped(clave_fin_2, clave_fin)
    parte_slowmo_c_procesada = parte_slowmo_c.with_effects([
        MultiplySpeed(factor=factor_velocidad),
        Blink(duration_on=duration_blink, duration_off=duration_blink)
    ])

    # parte 5: después del momento clave (velocidad normal)
    parte5 = clip.subclipped(clave_fin)

    # unir las partes
    clip_prepro = concatenate_videoclips(
        [parte1, parte_slowmo_a_procesada, parte_slowmo_b_procesada, parte_slowmo_c_procesada, parte5],
        method="compose",
    )
    # logger.info(f"parte1: fps={parte1.fps}, duration={parte1.duration}")
    # logger.info(f"parte_slowmo_procesada: fps={parte_slowmo_procesada.fps}, duration={parte_slowmo_procesada.duration}")
    # logger.info(f"parte5: fps={parte5.fps}, duration={parte5.duration}")
    # logger.info(f"clip_prepro: fps={clip_prepro.fps}, duration={clip_prepro.duration}")

    # NOTA SOBRE EL AUDIO: El slow motion alarga el vídeo pero no el audio.
    # Esta línea re-asigna el audio original al nuevo clip. El audio se desincronizará durante la parte lenta y se cortará si el vídeo final es más largo.
    # Para un resultado profesional, a menudo se añade música de fondo en su lugar.
    # if clip.audio:
    # clip_prepro = clip_prepro.with_audio(clip.audio)

    """clip_final = fn_slide(
        in_out="out",
        clip=clip_prepro,
        duration=0.5,
        slide=ut.random_efecto_slide(),
    )
    return clip_final"""

    return clip_prepro


def fn_slide(
        in_out: str,
        clip: VideoClip,
        duration: float = 2.0,
        slide: str = "top",  # 'top', 'bottom', 'left', 'right'
) -> CompositeVideoClip:
    # TODO por alguna razon no esta aplicando el efecto :(
    # video.fx.SlideIn: Hace que el clip entre en la pantalla deslizándose desde un lado.
    # video.fx.SlideOut: Hace que el clip salga de la pantalla deslizándose hacia un lado.

    if in_out == "in":
        clip_prepro = clip.with_effects([
            SlideIn(duration=duration, side=slide)
        ])

        clip_final = fn_slide(
            in_out="out",
            clip=clip_prepro,
            duration=0.5,
            slide=ut.random_efecto_slide(),
        )

        return clip_final
    elif in_out == "out":
        return clip.with_effects([
            SlideOut(duration=duration, side=slide)
        ])


def fn_freeze_region(
        clip: VideoClip,
        buffer: float = 1.5,
        porc_width: float = 0.5,
        porc_height: float = 0.5,
) -> CompositeVideoClip:
    # video.fx.FreezeRegion: Congela una parte específica de la pantalla mientras el resto del vídeo continúa reproduciéndose.

    # intuyendo que el evento clave ocurre en la mitad de la duracion del video
    clave_ini = clip.duration / 2 - buffer
    clave_fin = clip.duration / 2 + buffer
    parte_freeze = clip.subclipped(clave_ini, clave_fin)
    tt = parte_freeze.duration / 2

    w, h = clip.size
    x_center, y_center = w / 2, h / 2
    # rect_width, rect_height = 400, 300
    rect_width, rect_height = int(w * porc_width), int(h * porc_height)
    region_a_congelar = (
        int(x_center - rect_width / 2),
        int(y_center - rect_height / 2),
        int(x_center + rect_width / 2),
        int(y_center + rect_height / 2)
    )

    clip_prepro = parte_freeze.with_effects([
        FreezeRegion(t=tt, region=region_a_congelar)
    ])

    """clip_final = fn_slide(
        in_out="out",
        clip=clip_prepro,
        duration=0.5,
        slide=ut.random_efecto_slide(),
    )
    return clip_final"""

    return clip_prepro


def fn_time_mirror(
        clip: VideoClip,
        factor_velocidad: float = 1.5,
):
    # video.fx.TimeMirror: Reproduce el clip normalmente y luego lo reproduce en reversa, creando un efecto de espejo en el tiempo.

    # intuyendo que el evento clave ocurre en la mitad de la duracion del video
    clave_ini = clip.duration / 2
    parte_mirror = clip.subclipped(clave_ini)

    clip_prepro = parte_mirror.with_effects([
        MultiplySpeed(factor=factor_velocidad),
        TimeMirror()
    ])

    """clip_final = fn_slide(
        in_out="out",
        clip=clip_prepro,
        duration=0.5,
        slide=ut.random_efecto_slide(),
    )
    return clip_final"""

    return clip_prepro


def fn_blink(
        clip: VideoClip,
        duration_blink: float = 0.2,
        buffer: float = 1.5,
):
    # intuyendo que el evento clave ocurre en la mitad de la duracion del video
    clave_ini = clip.duration / 2 - buffer
    clave_fin = clip.duration / 2 + buffer

    parte1 = clip.subclipped(0, clave_ini)
    parte_blink = clip.subclipped(clave_ini, clave_fin).with_effects([
        Blink(duration_on=duration_blink, duration_off=duration_blink)
    ])
    parte3 = clip.subclipped(clave_fin)

    clip_prepro = concatenate_videoclips(
        [parte1, parte_blink, parte3],
        method="compose",
    )

    return clip_prepro


def video_resumen_moviepy(
        final_events,
        original_video_path: Path,
        buffer_seconds: float = 2.0,
):
    output_path = original_video_path.parent / f"{original_video_path.stem}_resumen_final.mp4"

    # Cargar el vídeo original una sola vez
    try:
        main_video_clip = VideoFileClip(str(original_video_path))
    except Exception as e:
        logger.error(f"Al cargar el vídeo original '{original_video_path}': {e}")
        return

    final_clips = []
    video_duration = main_video_clip.duration
    main_video_clip_w, main_video_clip_h = main_video_clip.size

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
            effect = None
            evento_con_efecto_duration = 0.0
            list_effects = []
            animate_clip_path = ""

            match evento_str:
                case "Goal":
                    effect = fn_multiply_speed_blink(clip=subclip, factor_velocidad=0.7)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    effect = fn_time_mirror(clip=subclip, factor_velocidad=1.5)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    effect = fn_freeze_region(clip=subclip, porc_width=0.5, porc_height=0.7)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    animate_clip_path = ut.random_video_file_clip(evento_clave=evento_str)
                    # logger.info(f"Goal - list_effects: {list_effects}")
                case "Yellow card":
                    effect = fn_multiply_speed_blink(clip=subclip, factor_velocidad=0.8)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    animate_clip_path = ut.random_video_file_clip(evento_clave=evento_str)
                    # logger.info(f"Yellow card - list_effects: {list_effects}")
                case "Substitution":
                    effect = fn_blink(clip=subclip, duration_blink=0.05)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    animate_clip_path = ut.random_video_file_clip(evento_clave=evento_str)
                    # logger.info(f"Substitution - list_effects: {list_effects}")
                case "Corner":
                    effect = subclip
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    effect = fn_freeze_region(clip=subclip, porc_width=0.5, porc_height=0.9)
                    evento_con_efecto_duration += effect.duration
                    list_effects.append(effect)

                    animate_clip_path = ut.random_video_file_clip(evento_clave=evento_str)
                    # logger.info(f"Corner - list_effects: {list_effects}")
                case _:
                    pass

            list_effects_concatenate = concatenate_videoclips(
                clips=list_effects,
                method="compose",
            )

            logger.info(f"Duración de {evento_str}: {evento_con_efecto_duration:.2f}s")
            txt_clip = TextClip(
                # font='Arial-Bold',
                # text=f"{evento_str}",
                text=f"{SOCCERNET_LABELS_ES.get(evento_str)}",
                font_size=120,
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
                duration=evento_con_efecto_duration,
            ).with_effects([
                CrossFadeIn(duration=2.0),
                CrossFadeOut(duration=2.0),
            ])

            # list_effects.append(txt_clip)
            # logger.info(f"TextClip - list_effects: {list_effects}")

            img_clip = ImageClip(
                img=Path(PATH_RESOURCES) / "cr7.png",
                transparent=True,
                duration=evento_con_efecto_duration,
            ).with_effects([
                Resize(height=main_video_clip_h * 0.35),
                CrossFadeIn(duration=2.0),
            ]).with_position(
                pos=("right", "bottom")
            )

            animate_clip = VideoFileClip(
                filename=Path(PATH_RESOURCES) / animate_clip_path,
                audio=False,
                has_mask=True,
            ).with_effects([
                Resize(height=main_video_clip_h * 0.25),
                CrossFadeIn(duration=1.5),
            ])
            # 'Loop' es muy sensible, por eso se separa
            animate_clip_2 = animate_clip.with_effects([
                Loop(duration=evento_con_efecto_duration),
            ]).with_position(
                pos=("right", "top")
            )

            list_f = [list_effects_concatenate, txt_clip, img_clip, animate_clip_2]

            effects_with_text = CompositeVideoClip(list_f).with_effects([
                FadeIn(duration=0.75),
                FadeOut(duration=0.75),
            ])

            final_clips.append(effects_with_text)
            # logger.info(f"final_clips: {final_clips}")
    except Exception as e:
        logger.error(e)

    if not final_clips:
        logger.error("No se pudo crear ningún subclip válido. Abortando la creación del vídeo.")
        main_video_clip.close()
        return

    try:
        # print(final_clips)
        logger.info(f"Concatenando todos los subclips en el video resumen final.")
        summary_video = concatenate_videoclips(
            clips=final_clips,
            # method="chain",
            method="compose",
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
