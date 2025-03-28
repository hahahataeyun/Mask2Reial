from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

img = Image.open("./graphene_image.png").convert("RGB")
img_tensor = transform(img)

# t_green = img_tensor[2, :, :]


# Denormalize
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


img_to_show = denormalize(t_green, mean=[0.5] * 3, std=[0.5] * 3)
img_np = img_to_show.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

# Show image
plt.imshow(img_np)
plt.title("Graphene Image")
plt.axis("off")
plt.show()
