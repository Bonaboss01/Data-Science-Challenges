from transformers import pipeline

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
result = captioner("my_image.jpg")
print(result)

from transformers import pipeline

def generate_caption(image_path):
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    result = captioner(image_path)
    return result[0]['generated_text']

print(generate_caption("image.jpg"))


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

print("Caption:", generate_caption("path_to_your_image.jpg"))


