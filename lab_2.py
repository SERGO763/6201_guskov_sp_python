import requests
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('api_key')
headers = {'x-api-key': api_key}
url = os.getenv('url')
r = requests.get(url, headers=headers)
response_dict = r.json()
breed = r.json()[0]['breeds'][0]['name']
image_url = response_dict[0]['url']
image_response = requests.get(image_url)
with open('photo.jpg', 'wb') as f:
    f.write(image_response.content)

image = Image.open('photo.jpg').convert('RGB')
image_array = np.array(image)
kernel = np.array([[1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9]])

def convolve(image_array, kernel):
    """Функция для выполнения свертки"""
    image_array_1 = np.zeros((image_array.shape[0] + 2, image_array.shape[1] + 2, image_array.shape[2]), dtype=image_array.dtype)
    image_array_1[1: - 1, 1: - 1, :] = image_array[:, :, :]

    img_height, img_width, num_channels = image_array_1.shape
    kernel_height, kernel_width = kernel.shape

    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    output = np.zeros((output_height, output_width, num_channels))
    for channel in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                region = image_array_1[i:i + kernel_height, j:j + kernel_width, channel]
                output[i, j, channel] = np.sum(region * kernel)

    output = np.clip(output, 0, 255)
    output /= 255.0

    max_red = np.max(output[:, :, 0])
    max_blue = np.max(output[:, :, 1])
    max_green = np.max(output[:, :, 2])

    min_red = np.min(output[:, :, 0])
    min_green = np.min(output[:, :, 1])
    min_blue = np.min(output[:, :, 2])

    k_red = 255 / (max_red - min_red)
    k_green = 255 / (max_green - min_green)
    k_blue = 255 / (max_blue - min_blue)

    b_red = 0 - (k_red * min_red)
    b_green = 0 - (k_green * min_green)
    b_blue = 0 - (k_blue * min_blue)

    output[:, :, 0] = (k_red * output[:, :, 0] + b_red)
    output[:, :, 1] = (k_green * output[:, :, 1] + b_green)
    output[:, :, 2] = (k_blue * output[:, :, 2] + b_blue)

    output = output.astype(np.uint8)
    result_image = Image.fromarray(output)
    result_image.save(f'{breed}_1.jpg')

    return output

result_2 = np.zeros_like(image_array)
for channel in range(image_array.shape[2]):
    result_2[:, :, channel] = convolve2d(image_array[:, :, channel], kernel, mode='same', boundary='wrap')
plt.imsave(f'{breed}_2.jpg', result_2)
convolve(image_array, kernel)








