from moviepy import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import Resize, FadeIn, FadeOut, CrossFadeIn, CrossFadeOut

# Cargar clips
clip1 = VideoFileClip("pocs/clip1.mkv").subclipped(0, 3)
clip2 = VideoFileClip("pocs/clip2.mkv").subclipped(0, 3)

# Crear texto con fade
text = TextClip(text="¡GOL!", font_size=70, color='yellow', method='label',duration=clip1.duration)
text = text.with_effects([FadeIn(1.0)])

# Ícono balón (PNG)
ball = (
    ImageClip("ball.png")
    .set_duration(3)
    .set_position(("left", "top"))
    .fx(FadeIn, 0.5)
    .fx(Resize, height=60)
)

# Ligerísimo zoom al fondo del clip1
clip1_zoom = clip1.fx(Resize, 1.05)

# Superposición de elementos visuales sobre clip1
clip1_final = CompositeVideoClip([clip1_zoom, text, ball])

# Aplicar crossfade entre clips (compatible)
clip1_final = clip1_final.fx(CrossFadeOut, 1)
clip2 = clip2.fx(CrossFadeIn, 1)

# Concatenar
final_video = concatenate_videoclips([clip1_final, clip2], method="compose")

# Exportar
final_video.write_videofile("videoresumen_pro.mp4", codec="libx264", audio_codec="aac")
"""