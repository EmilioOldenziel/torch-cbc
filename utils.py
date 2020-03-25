import cv2 as cv


def visualize_components(epoch, model):
    for idx, c in enumerate(model.components):
        component = c
        img = component.view(28, 28).cpu().data.numpy()
        img = img * 255
        img = cv.resize(img, (56, 56))
        cv.imwrite(f"char_img/{epoch}_{idx}.png", img)
