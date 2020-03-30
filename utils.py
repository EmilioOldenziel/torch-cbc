import os
from PIL import Image, ImageOps


def visualize_components(epoch, model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, c in enumerate(model.components):
        component = c
        img = component.view(28, 28).cpu().data.numpy()
        img = img * 255

        image = Image.fromarray(img).convert('RGB')
        image = image.resize((56, 56))
        img_with_border.save(f"{save_path}/{epoch}_{idx}.png")
