import requests
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from dotenv import load_dotenv
import os
from numba import njit

class ImageProcessor:
    def __init__(self, breed, image_array):
        self.__breed = breed
        self.__image = image_array

    @classmethod
    def download_image(cls):
        """Функция, которая подключается к API и загружает до 3 изображений."""
        images_directory = 'images'
        os.makedirs(images_directory, exist_ok=True)
        load_dotenv()
        api_key = os.getenv('api_key')
        headers = {'x-api-key': api_key}
        url = os.getenv('url')
        r = requests.get(url, headers=headers)
        response_dict = r.json()
        breed = response_dict[0]['breeds'][0]['name']
        print(f"Загруженная порода: {breed}")

        num_images = min(3, len(response_dict))
        image_paths = []
        for i in range(num_images):
            image_url = response_dict[i]['url']
            image_response = requests.get(image_url)
            image_path = os.path.join(images_directory, f'{breed}.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_response.content)
        image_array = np.array(Image.open(image_path).convert('RGB'))

        image_paths.append(image_path)
        return cls(breed, image_array)


    def _convolve(self, kernel):
        """Функция для выполнения свертки"""
        images_directory = 'images'
        os.makedirs(images_directory, exist_ok=True)
        for i in range(3):
            image_array_1 = np.zeros((self.__image.shape[0] + 2, self.__image.shape[1] + 2, self.__image.shape[2]),
                                    dtype=self.__image.dtype)
            image_array_1[1: - 1, 1: - 1, :] = self.__image[:, :, :]

            img_height, img_width, num_channels = image_array_1.shape
            kernel_height, kernel_width = kernel.shape

            output_height = img_height - kernel_height + 1
            output_width = img_width - kernel_width + 1

            output = np.zeros((output_height, output_width, num_channels))
            convolve_numba(num_channels, output_height, output_width, image_array_1, kernel_height, kernel_width, output)

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
            result_image.save(os.path.join(images_directory, f'{self.__breed}_convolve.jpg'))
        return output

    def _scipy_convolve(self, kernel):
        """Функция для выполнения свертки через Scipy"""
        for i in range(3):
            result_2 = np.zeros_like(self.__image)

            for channel in range(self.__image.shape[2]):
                result_2[:, :, channel] = convolve2d(self.__image[:, :, channel], kernel, mode='same', boundary='wrap')

            result_2 = np.clip(result_2, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(result_2)

            result_image.save(os.path.join('images', f'{self.__breed}_scipy.jpg'))

@njit
def convolve_numba(num_channels, output_height, output_width, image_array_1, kernel_height, kernel_width, output):
    for channel in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                region = image_array_1[i:i + kernel_height, j:j + kernel_width, channel]
                output[i, j, channel] = np.sum(region * kernel)

kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9],
                [1 / 9, 1 / 9, 1 / 9]])

for x in range(3):
    processor = ImageProcessor.download_image()
    processor._convolve(kernel)
    processor._scipy_convolve(kernel)
