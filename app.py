import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/mo-di-diffusion",
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda" if torch.cuda.is_available() else "cpu")

def toonify(image, prompt_extra):
    prompt = f"Cartoon portrait, {prompt_extra}"
    image = image.convert("RGB").resize((512, 512))
    output = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5)
    return output.images[0]

gr.Interface(
    fn=toonify,
    inputs=[
        gr.Image(type="pil", label="Imagen de entrada"),
        gr.Textbox(label="Descripción adicional (opcional)", placeholder="e.g. cycling with sunglasses"),
    ],
    outputs=gr.Image(label="Caricatura generada"),
    title="🖼️ Toonify with Stable Diffusion",
    description="Genera una caricatura a partir de una imagen real. Usando el modelo mo-di-diffusion de nitrosocke."
).launch()
