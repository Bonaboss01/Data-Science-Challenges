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

