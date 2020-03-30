import os
from PIL import Image, ImageOps


def get_concat_h(images):
    dst = Image.new('RGB', (9*(56+8), 56+8))
    for i, im in enumerate(images):
        dst.paste(im, (i*(56+8), 0))
    return dst


def visualize_components(epoch, model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = []
    for idx, c in enumerate(model.components):
        component = c
        img = component.view(28, 28).cpu().data.numpy()
        img = img * 255

        image = Image.fromarray(img).convert('RGB')
        image = image.resize((56, 56))
        # image.save(f"{save_path}/{epoch}_{idx}.png")

        image = ImageOps.expand(image, border=4, fill='black')
        images.append(image)

    result = get_concat_h(images)
    result.save(f"{save_path}/{epoch}.png")
