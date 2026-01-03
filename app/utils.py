import io
import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes): 
    #define the same transforms as training, but add resize
    my_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #open the image file 
    image = Image.open(io.BytesIO(image_bytes))

    #ensure its rgb
    if image.mode != "RGB": 
        image = image.convert("RGB")

    #apply transforms and ass batch dimension (1, 3, 32, 32)
    return my_transforms(image).unsqueeze(0)
