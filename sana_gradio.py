import torch
import gradio as gr
from diffusers import SanaPAGPipeline
import random

pipe = SanaPAGPipeline.from_pretrained(
    "kpsss34/SANA600.fp16_Realistic_SFW_V1",
    torch_dtype=torch.bfloat16,
    pag_applied_layers="transformer_blocks.8",
)
pipe.to("cuda")
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

def infer(prompt):
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=5.0,
        pag_scale=4.0,
        num_inference_steps=25,
        generator=generator,
    )[0][0]
    return image, f"Seed: {seed}"

gr.Interface(
    fn=infer,
    inputs=gr.Textbox(label="Prompt", placeholder="you prompt dude, Have fun!"),
    outputs=[
        gr.Image(type="pil", label="Result Image"),
        gr.Textbox(label="Random Seed Used")
    ],
    title="SANA PAG - Local Infer",
    description="Type a prompt to have the SANA model generate a random Seed image every time.",
    allow_flagging="never"
).launch(server_name="127.0.0.1", server_port=7860, share=True)
