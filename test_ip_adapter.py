import torch
from diffusers import DiffusionPipeline
import torchvision

from ip_adapter import IPAdapter

model_path = "damo-vilab/text-to-video-ms-1.7b"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
step=3000
ip_ckpt = f"/fs/nexus-scratch/vatsalb/runs/mrbean_look_ipadapter/checkpoint-{step}.bin"
device = "cuda"

pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
video, audio, fps = torchvision.io.read_video("/nfshomes/vatsalb/videos/mrbean_look.mp4", pts_unit='sec')
image = torchvision.transforms.functional.to_pil_image(video[0].permute(2, 0, 1))
image.resize((256, 256))
images = ip_model.generate(pil_image=image, scale=0, num_samples=1, num_inference_steps=25, seed=42, prompt="A man wearing brown suit in grassy field")

output = torch.stack([torch.Tensor(frame) for frame in images.frames])
torchvision.io.write_video(f"ip_output_{step}.mp4", output, fps=8)