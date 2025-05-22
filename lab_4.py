import requests
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from dotenv import load_dotenv
import os
from numba import njit


class ImagePipeline:
    def __init__(self):
        self.images_directory = 'processed_images'
        os.makedirs(self.images_directory, exist_ok=True)
        load_dotenv()
        self.api_key = os.getenv('api_key')
        self.url = os.getenv('url')
        self.kernel = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])

    def download_images(self):
        """Генератор для скачивания изображений"""
        headers = {'x-api-key': self.api_key}
        response = requests.get(self.url, headers=headers)
        response_dict = response.json()

        for idx, image_data in enumerate(response_dict[:3], start=1):
            breed = image_data['breeds'][0]['name']
            image_url = image_data['url']
            image_response = requests.get(image_url)

            original_path = os.path.join(self.images_directory, f'{idx}_{breed}_original.jpg')
            with open(original_path, 'wb') as f:
                f.write(image_response.content)

            image_array = np.array(Image.open(original_path).convert('RGB'))
            yield idx, breed, image_array

    def apply_custom_filter(self, image_stream):
        """Генератор для применения фильтра"""
        for idx, breed, image_array in image_stream:
            filtered_image = self._convolve(image_array)
            custom_path = os.path.join(self.images_directory, f'{idx}_{breed}_convolve.jpg')
            Image.fromarray(filtered_image).save(custom_path)
            yield idx, breed, custom_path

    def apply_scipy_filter(self, image_stream):
        """Генератор для применения scipy фильтра"""
        for idx, breed, _ in image_stream:
            image_path = os.path.join(self.images_directory, f'{idx}_{breed}_original.jpg')
            image_array = np.array(Image.open(image_path).convert('RGB'))

            filtered_image = self._scipy_convolve(image_array)
            scipy_path = os.path.join(self.images_directory, f'{idx}_{breed}_scipy.jpg')
            Image.fromarray(filtered_image).save(scipy_path)
            yield idx, breed, scipy_path

    def _convolve(self, image_array):
        """Метод для выполнения свертки"""
        pad_width = ((1, 1), (1, 1), (0, 0))
        image_array_1 = np.pad(image_array, pad_width, mode='edge')

        img_height, img_width, num_channels = image_array_1.shape
        kernel_height, kernel_width = self.kernel.shape

        output_height = img_height - kernel_height + 1
        output_width = img_width - kernel_width + 1

        output = np.zeros((output_height, output_width, num_channels))
        self._convolve_numba(image_array_1, output)

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    def _scipy_convolve(self, image_array):
        """Метод для выполнения свертки через Scipy"""
        result = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            result[:, :, channel] = convolve2d(image_array[:, :, channel],
                                               self.kernel, mode='same', boundary='symm')
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    @njit
    def _convolve_numba(image_array_1, output):
        """Numba-оптимизированная свертка"""
        kernel = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])
        num_channels = image_array_1.shape[2]
        kernel_height, kernel_width = kernel.shape

        for channel in range(num_channels):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    region = image_array_1[i:i + kernel_height, j:j + kernel_width, channel]
                    output[i, j, channel] = np.sum(region * kernel)


def run_pipeline():
    pipeline = ImagePipeline()

    # Создаем пайплайн
    download_gen = pipeline.download_images()
    custom_filter_gen = pipeline.apply_custom_filter(download_gen)
    scipy_filter_gen = pipeline.apply_scipy_filter(custom_filter_gen)

    for idx, breed, scipy_path in scipy_filter_gen:
        print(f"Обработка завершена для изображения {idx} (порода: {breed})")
        print(f"Результат сохранен в: {scipy_path}\n")

    print("Все изображения успешно обработаны!")


if __name__ == "__main__":
    run_pipeline()
