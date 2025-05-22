import argparse
import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
from functools import partial

import aiofiles
import aiohttp
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.signal import convolve2d
from numba import njit


class CatImageProcessor:
    def __init__(self, output_dir: str = 'processed_images'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        load_dotenv()
        self.api_key = os.getenv('api_key')
        self.base_url = os.getenv('url')
        self.kernel = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])
        # Используем максимальное количество доступных ядер
        self.process_pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.process_pool.shutdown()

    async def fetch_image_urls(self, limit: Optional[int] = None) -> List[Tuple[int, str, str]]:
        """Асинхронно получает список URL изображений и пород"""
        headers = {'x-api-key': self.api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, headers=headers) as response:
                data = await response.json()
                if limit:
                    data = data[:limit]

                return [
                    (idx + 1,
                     img['breeds'][0]['name'] if img.get('breeds') else 'unknown',
                     img['url'])
                    for idx, img in enumerate(data)
                ]
    @staticmethod
    async def download_image(output_dir, session: aiohttp.ClientSession,
                           idx: int, breed: str, url: str) -> Tuple[int, str, str]:
        """Асинхронно скачивает и сохраняет изображение"""
        print(f'Downloading image {idx} started')
        async with session.get(url) as response:
            original_path = os.path.join(output_dir, f'{idx}_{breed}_original.jpg')
            async with aiofiles.open(original_path, 'wb') as f:
                await f.write(await response.read())

        print(f'Downloading image {idx} finished')
        return idx, breed, original_path

    @staticmethod
    def _apply_custom_convolution_static(image_path: str, output_dir: str) -> Tuple[str, np.ndarray]:
        """Статический метод для применения кастомной свертки"""
        print(f'Custom convolution process started (PID {os.getpid()})')
        image_array = np.array(Image.open(image_path).convert('RGB'))
        pad_width = ((1, 1), (1, 1), (0, 0))
        padded = np.pad(image_array, pad_width, mode='edge')

        output = np.zeros((padded.shape[0] - 2, padded.shape[1] - 2, 3))
        CatImageProcessor._convolve_numba(padded, output)

        result = np.clip(output, 0, 255).astype(np.uint8)
        return os.path.basename(image_path).replace('_original.jpg', '_convolve.jpg'), result

    @staticmethod
    def _apply_scipy_convolution_static(image_path: str, output_dir: str) -> Tuple[str, np.ndarray]:
        """Статический метод для применения scipy свертки"""
        print(f'Scipy convolution process started (PID {os.getpid()})')
        img = np.array(Image.open(image_path).convert('RGB'))
        result = np.zeros_like(img)
        kernel = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])

        for channel in range(img.shape[2]):
            result[:, :, channel] = convolve2d(
                img[:, :, channel],
                kernel,
                mode='same',
                boundary='symm'
            )

        result = np.clip(result, 0, 255).astype(np.uint8)
        return os.path.basename(image_path).replace('_original.jpg', '_scipy.jpg'), result

    @staticmethod
    @njit
    def _convolve_numba(padded: np.ndarray, output: np.ndarray):
        """Numba-оптимизированная свертка"""
        kernel = np.array([
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])
        for channel in range(3):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    region = padded[i:i + 3, j:j + 3, channel]
                    output[i, j, channel] = np.sum(region * kernel)

    async def process_single_image(self, idx: int, breed: str, image_path: str) -> Tuple[int, str, Tuple[str, str]]:
        """Обрабатывает одно изображение с применением обоих фильтров параллельно"""
        loop = asyncio.get_running_loop()

        custom_func = partial(self._apply_custom_convolution_static, output_dir=self.output_dir)
        scipy_func = partial(self._apply_scipy_convolution_static, output_dir=self.output_dir)

        # Запускаем оба фильтра параллельно
        custom_task = loop.run_in_executor(self.process_pool, custom_func, image_path)
        scipy_task = loop.run_in_executor(self.process_pool, scipy_func, image_path)


        (custom_name, custom_image), (scipy_name, scipy_image) = await asyncio.gather(custom_task, scipy_task)

        # Сохраняем результаты
        custom_path = os.path.join(self.output_dir, custom_name)
        scipy_path = os.path.join(self.output_dir, scipy_name)

        Image.fromarray(custom_image).save(custom_path)
        Image.fromarray(scipy_image).save(scipy_path)

        print(f'Processing complete for image {idx}')
        return idx, breed, (custom_path, scipy_path)

    async def process_pipeline(self, limit: Optional[int] = None):
        start_time = time.time()

        image_infos = await self.fetch_image_urls(limit)

        async with aiohttp.ClientSession() as session:
            # Скачиваем все изображения параллельно
            download_tasks = [
                self.download_image('processed_images', session, idx, breed, url)
                for idx, breed, url in image_infos
            ]
            downloaded_images = await asyncio.gather(*download_tasks)

        process_tasks = [
            self.process_single_image(idx, breed, image_path)
            for idx, breed, image_path in downloaded_images
        ]
        results = await asyncio.gather(*process_tasks)

        for idx, breed, (custom_path, scipy_path) in results:
            print(f"Обработка завершена для изображения {idx} (порода: {breed})")
            print(f"Результаты сохранены в: {custom_path} и {scipy_path}\n")

        end_time = time.time()
        print(f"Все изображения успешно обработаны! Время выполнения: {end_time - start_time:.2f} секунд")


async def main(limit: Optional[int] = None):
    async with CatImageProcessor() as processor:
        await processor.process_pipeline(limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Асинхронная обработка изображений кошек')
    parser.add_argument('--limit', type=int, help='Ограничение количества обрабатываемых изображений')
    args = parser.parse_args()

    asyncio.run(main(args.limit))
