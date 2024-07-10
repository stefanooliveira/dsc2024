import cv2
import numpy as np
from matplotlib import pyplot as plt

def crop_map_limits(image_path, x0, y0, x1, y1, save_path=None):
    image = cv2.imread(image_path)
    cropped_image = image[y0:y1, x0:x1]

    if save_path:
        cv2.imwrite(save_path, cropped_image)

    return cropped_image

image_path = 'S11635384_202206010100.jpg'
cropped_image_path = 'cropped_map.jpg'

x0, y0, x1, y1 = 45, 107, 2174, 2235

cropped_image = crop_map_limits(image_path, x0, y0, x1, y1, cropped_image_path)

original_image = cv2.imread(image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Imagem Recortada")
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

plt.show()
