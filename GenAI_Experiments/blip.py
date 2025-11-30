from transformers import pipeline

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
result = captioner("my_image.jpg")
print(result)
