import sys
import os
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel,
                            QPushButton, QComboBox, QHBoxLayout, QVBoxLayout,
                            QWidget, QProgressBar, QMessageBox, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable
from numba import njit, prange, set_num_threads, get_num_threads
import math
from functools import lru_cache
import concurrent.futures
from multiprocessing import cpu_count

# Установка количества потоков поумолчанию
MAX_THREADS = max(1, cpu_count() - 1)
set_num_threads(MAX_THREADS)

# Темная тема
DARK_STYLE = """
QMainWindow {
    background-color: #2D2D2D;
    color: #E0E0E0;
    font-family: Arial, sans-serif;
    font-size: 12px;
}

QLabel {
    color: #E0E0E0;
    qproperty-alignment: AlignCenter;
    border: 2px dashed #555;
    background-color: #252525;
    padding: 10px;
}

QPushButton {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #4A4A4A;
}

QPushButton:disabled {
    background-color: #2A2A2A;
    color: #707070;
}

QComboBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
    min-width: 150px;
}

QComboBox QAbstractItemView {
    background-color: #3A3A3A;
    color: #E0E0E0;
    selection-background-color: #505050;
}

QProgressBar {
    border: 1px solid #555;
    border-radius: 4px;
    text-align: center;
    background-color: #252525;
    color: #E0E0E0;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4CAF50;
    border-radius: 3px;
}

QMessageBox {
    background-color: #2D2D2D;
}

QMessageBox QLabel {
    color: #E0E0E0;
    border: none;
    background-color: transparent;
}

QSpinBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
    min-width: 60px;
}
"""

# Константы для преобразования цветов
D65_X = 95.047
D65_Y = 100.000
D65_Z = 108.883
CIE_E = 216.0 / 24389.0
CIE_K = 24389.0 / 27.0

# Коэффициенты для Weighted Euclidean
WEIGHTED_EUCLIDEAN_WEIGHTS = np.array([0.299, 0.587, 0.114])

# Коэффициенты для Rec. ITU-R BT.2124 (2019)
BT2124_COEFFS = np.array([
    [0.70, 0.30, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00]
])

@lru_cache(maxsize=65536)
def rgb_to_lab_cached(rgb_tuple):
    """Кэшированное преобразование RGB в Lab"""
    return rgb_to_lab_numba(np.array(rgb_tuple))

@njit(fastmath=True)
def rgb_to_xyz_numba(rgb):
    """Преобразование RGB в XYZ с использованием Numba"""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    # Inverse sRGB companding
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

    # D65 reference white
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x * 100.0, y * 100.0, z * 100.0

@njit(fastmath=True)
def xyz_to_lab_numba(xyz):
    """Преобразование XYZ в Lab с использованием Numba"""
    x, y, z = xyz
    # D65 reference white
    x /= D65_X
    y /= D65_Y
    z /= D65_Z

    # Nonlinear transform
    x = x ** (1/3) if x > CIE_E else (CIE_K * x + 16) / 116
    y = y ** (1/3) if y > CIE_E else (CIE_K * y + 16) / 116
    z = z ** (1/3) if z > CIE_E else (CIE_K * z + 16) / 116

    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return l, a, b

@njit(fastmath=True)
def rgb_to_lab_numba(rgb):
    """Преобразование RGB в Lab с использованием Numba"""
    xyz = rgb_to_xyz_numba(rgb)
    return xyz_to_lab_numba(xyz)

@njit(fastmath=True)
def ciede2000_numba_single(lab1, lab2):
    """Оптимизированная версия CIEDE2000 для одиночных цветов"""
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # Вычисление C'
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) * 0.5

    # Расчет G
    C_avg_pow7 = C_avg**7
    G = 0.5 * (1.0 - math.sqrt(C_avg_pow7 / (C_avg_pow7 + 6103515625.0)))  # 25^7 = 6103515625

    # a' вычисления
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    # Углы h'
    h1_prime = 0.0 if (b1 == 0.0 and a1_prime == 0.0) else math.atan2(b1, a1_prime)
    if h1_prime < 0.0:
        h1_prime += 2.0 * math.pi

    h2_prime = 0.0 if (b2 == 0.0 and a2_prime == 0.0) else math.atan2(b2, a2_prime)
    if h2_prime < 0.0:
        h2_prime += 2.0 * math.pi

    # Разницы
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Вычисление delta_h_prime
    if C1_prime * C2_prime == 0.0:
        delta_h_prime = 0.0
    elif abs(h2_prime - h1_prime) <= math.pi:
        delta_h_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > math.pi:
        delta_h_prime = h2_prime - h1_prime - 2.0 * math.pi
    else:
        delta_h_prime = h2_prime - h1_prime + 2.0 * math.pi

    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime * 0.5)

    # Средние значения
    L_avg_prime = (L1 + L2) * 0.5
    C_avg_prime = (C1_prime + C2_prime) * 0.5

    # Вычисление h_avg_prime
    if C1_prime * C2_prime == 0.0:
        h_avg_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= math.pi:
            h_avg_prime = (h1_prime + h2_prime) * 0.5
        elif h1_prime + h2_prime < 2.0 * math.pi:
            h_avg_prime = (h1_prime + h2_prime + 2.0 * math.pi) * 0.5
        else:
            h_avg_prime = (h1_prime + h2_prime - 2.0 * math.pi) * 0.5

    # Весовые коэффициенты
    T = (1.0 - 0.17 * math.cos(h_avg_prime - 0.5235987755982988) +  # pi/6
         0.24 * math.cos(2.0 * h_avg_prime) +
         0.32 * math.cos(3.0 * h_avg_prime + 0.10471975511965977) -  # pi/30
         0.20 * math.cos(4.0 * h_avg_prime - 1.0995574287564276))    # 63*pi/180

    delta_theta = 0.5235987755982988 * math.exp(-((h_avg_prime * 57.29577951308232 - 275.0)/25.0)**2)

    R_C = 2.0 * math.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 6103515625.0))
    S_L = 1.0 + (0.015 * (L_avg_prime - 50.0)**2) / math.sqrt(20.0 + (L_avg_prime - 50.0)**2)
    S_C = 1.0 + 0.045 * C_avg_prime
    S_H = 1.0 + 0.015 * C_avg_prime * T

    R_T = -math.sin(2.0 * delta_theta) * R_C

    # Итоговое delta_E
    delta_E = math.sqrt((delta_L_prime / S_L)**2 +
                       (delta_C_prime / S_C)**2 +
                       (delta_H_prime / S_H)**2 +
                       R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))

    return delta_E

@njit(fastmath=True, parallel=True)
def ciede2000_numba_batch(lab_pixels, lab_palette):
    """Векторизованная версия CIEDE2000 для пакетной обработки"""
    n_pixels = lab_pixels.shape[0]
    n_colors = lab_palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = ciede2000_numba_single(lab_pixels[i], lab_palette[j])

    return distances

@njit(fastmath=True)
def weighted_euclidean_distance(rgb1, rgb2, weights):
    """Вычисление взвешенного евклидова расстояния"""
    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]

    return math.sqrt(weights[0] * r_diff**2 +
                    weights[1] * g_diff**2 +
                    weights[2] * b_diff**2)

@njit(fastmath=True, parallel=True)
def weighted_euclidean_batch(pixels, palette, weights):
    """Векторизованная версия взвешенного евклидова расстояния"""
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = weighted_euclidean_distance(pixels[i], palette[j], weights)

    return distances

@njit(fastmath=True)
def bt2124_transform(rgb):
    """Преобразование RGB в цветовое пространство Rec. ITU-R BT.2124"""
    r = rgb[0] / 255.0
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    # Применяем матрицу преобразования
    c1 = BT2124_COEFFS[0,0] * r + BT2124_COEFFS[0,1] * g + BT2124_COEFFS[0,2] * b
    c2 = BT2124_COEFFS[1,0] * r + BT2124_COEFFS[1,1] * g + BT2124_COEFFS[1,2] * b
    c3 = BT2124_COEFFS[2,0] * r + BT2124_COEFFS[2,1] * g + BT2124_COEFFS[2,2] * b

    return c1, c2, c3

@njit(fastmath=True)
def bt2124_distance(rgb1, rgb2):
    """Вычисление расстояния в цветовом пространстве Rec. ITU-R BT.2124"""
    c1_1, c2_1, c3_1 = bt2124_transform(rgb1)
    c1_2, c2_2, c3_2 = bt2124_transform(rgb2)

    delta_c1 = c1_1 - c1_2
    delta_c2 = c2_1 - c2_2
    delta_c3 = c3_1 - c3_2

    return math.sqrt(delta_c1**2 + delta_c2**2 + delta_c3**2)

@njit(fastmath=True, parallel=True)
def bt2124_batch(pixels, palette):
    """Векторизованная версия расстояния Rec. ITU-R BT.2124"""
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = bt2124_distance(pixels[i], palette[j])

    return distances

class ImageConverter(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    file_finished = pyqtSignal(str)

    def __init__(self, image_path, palette_colors, method='euclidean', block_size=1000, threads=MAX_THREADS):
        super().__init__()
        self.image_path = image_path
        self.palette_colors = palette_colors
        self.method = method
        self.block_size = block_size
        self.threads = threads
        self._is_running = True
        set_num_threads(threads)

    def run(self):
        try:
            # Загрузка изображения
            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.uint8)
            height, width = img_array.shape[:2]

            # Преобразуем палитру в массив numpy
            palette_array = np.array([(int(c[2:4],16), int(c[4:6],16), int(c[6:8],16)) for c in self.palette_colors],
                                   dtype=np.uint8)

            if self.method == 'euclidean':
                # Многопоточная реализация евклидова расстояния
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_euclidean_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            elif self.method == 'ciede2000_optimized':
                pixels = img_array.reshape(-1, 3)
                palette_lab = np.array([rgb_to_lab_numba(c) for c in palette_array])

                result = np.zeros((height, width, 3), dtype=np.uint8)
                total_pixels = pixels.shape[0]
                processed_pixels = 0

                # Обрабатываем изображение блоками с использованием ThreadPool
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, self.block_size):
                        if not self._is_running:
                            return

                        block_end = min(i + self.block_size, total_pixels)
                        block = pixels[i:block_end]

                        futures.append(executor.submit(
                            self.process_block,
                            block, palette_lab, palette_array, i, width
                        ))

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        block_result, block_start, block_processed = future.result()

                        # Заполняем результат
                        for j in range(block_result.shape[0]):
                            pixel_idx = block_start + j
                            y = pixel_idx // width
                            x = pixel_idx % width
                            result[y, x] = block_result[j]

                        # Обновляем прогресс
                        processed_pixels += block_processed
                        progress = int((processed_pixels / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                self.result_ready.emit(result)

            elif self.method == 'weighted_euclidean':
                # Реализация взвешенного евклидова расстояния
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_weighted_euclidean_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                WEIGHTED_EUCLIDEAN_WEIGHTS,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            elif self.method == 'bt2124':
                # Реализация Rec. ITU-R BT.2124 (2019)
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_bt2124_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            # Сохраняем результат
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_converted.png")
            Image.fromarray(result, 'RGB').save(output_path)

            self.file_finished.emit(output_path)

        except Exception as e:
            print(f"Error in ImageConverter: {e}")
        finally:
            self.finished.emit()

    def process_euclidean_chunk(self, pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием евклидова расстояния"""
        # Создаём дерево для быстрого поиска ближайших цветов
        tree = cKDTree(palette_array)

        # Находим ближайшие цвета для всех пикселей в блоке
        distances, indices = tree.query(pixels_chunk, workers=1)

        # Получаем цвета из палитры по найденным индексам
        chunk_result = palette_array[indices]

        return chunk_result, start_idx, end_idx

    def process_weighted_euclidean_chunk(self, pixels_chunk, palette_array, weights, start_idx, end_idx):
        """Обработка блока пикселей с использованием взвешенного евклидова расстояния"""
        # Вычисляем расстояния
        distances = weighted_euclidean_batch(pixels_chunk, palette_array, weights)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx

    def process_bt2124_chunk(self, pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием Rec. ITU-R BT.2124"""
        # Вычисляем расстояния
        distances = bt2124_batch(pixels_chunk, palette_array)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx

    def process_block(self, block, palette_lab, palette_array, block_start, image_width):
        """Обработка блока пикселей в отдельном потоке (для CIEDE2000)"""
        # Преобразуем блок в Lab
        block_lab = np.array([rgb_to_lab_numba(p) for p in block])

        # Находим ближайшие цвета в палитре
        distances = ciede2000_numba_batch(block_lab, palette_lab)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        block_result = palette_array[best_indices]

        return block_result, block_start, block.shape[0]

    def stop(self):
        self._is_running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.palettes = {}
        self.setAcceptDrops(True)
        self.conversion_queue = []
        self.is_processing_queue = False
        self.current_palette = None
        self.current_image_path = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.setup_ui()
        self.setStyleSheet(DARK_STYLE)
        self.load_palettes_on_startup()

    def setup_ui(self):
        self.setWindowTitle("Mapart Helper")
        self.setGeometry(100, 100, 1000, 650)

        # Создаем главный виджет и основной layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Layout для изображений (горизонтальный)
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        # Исходное изображение
        self.image_label = QLabel("Перетащите изображение сюда")
        self.image_label.setMinimumSize(450, 400)
        images_layout.addWidget(self.image_label)

        # Результат
        self.result_label = QLabel("Результат конвертации")
        self.result_label.setMinimumSize(450, 400)
        images_layout.addWidget(self.result_label)

        main_layout.addLayout(images_layout)

        # Панель управления
        control_panel = QHBoxLayout()
        control_panel.setSpacing(10)

        self.open_image_btn = QPushButton("Открыть изображение")
        self.open_palette_btn = QPushButton("Открыть палитру")
        self.convert_btn = QPushButton("Конвертировать")
        self.convert_btn.setEnabled(False)

        control_panel.addWidget(self.open_image_btn)
        control_panel.addWidget(self.open_palette_btn)
        control_panel.addWidget(self.convert_btn)

        main_layout.addLayout(control_panel)

        # Выбор палитры и метода
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)

        settings_layout.addWidget(QLabel("Палитра:"))

        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumWidth(200)
        settings_layout.addWidget(self.palette_combo)

        settings_layout.addWidget(QLabel("Метод:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Евклидово расстояние (быстро)",
            "CIEDE2000 (оптимизированная)",
            "Взвешенное евклидово",
            "Rec. ITU-R BT.2124 (2019)"
        ])
        self.method_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.method_combo)

        # Настройки потоков
        settings_layout.addWidget(QLabel("Потоки:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, MAX_THREADS)
        self.threads_spin.setValue(MAX_THREADS)
        settings_layout.addWidget(self.threads_spin)

        main_layout.addLayout(settings_layout)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Подключаем сигналы
        self.open_image_btn.clicked.connect(self.open_image)
        self.open_palette_btn.clicked.connect(self.open_palette)
        self.convert_btn.clicked.connect(self.convert_current_image)
        self.palette_combo.currentTextChanged.connect(self.select_palette)

    def load_palettes_on_startup(self):
        """Загружает все сохранённые палитры при старте"""
        palettes_dir = "palettes"
        if not os.path.exists(palettes_dir):
            os.makedirs(palettes_dir)
            return

        for file in os.listdir(palettes_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(palettes_dir, file), 'r') as f:
                        palette_name = os.path.splitext(file)[0]
                        self.palettes[palette_name] = json.load(f)
                except Exception as e:
                    print(f"Ошибка загрузки палитры {file}: {e}")

        self.update_palette_combo()

    def update_palette_combo(self):
        self.palette_combo.clear()
        self.palette_combo.addItem("-- Выберите палитру --")
        self.palette_combo.addItems(sorted(self.palettes.keys()))

    def open_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")

        if paths:
            if len(paths) == 1:
                self.load_image(paths[0])
            else:
                self.add_files_to_queue(paths)
                self.convert_btn.setEnabled(self.current_palette is not None)

    def load_image(self, path):
        try:
            self.current_image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.convert_btn.setEnabled(self.current_palette is not None)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def open_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл палитры", "",
            "Palette files (*.txt *.json)")
        if path:
            self.load_palette_file(path)

    def load_palette_file(self, path):
        try:
            if path.lower().endswith('.txt'):
                with open(path, 'r') as f:
                    lines = f.readlines()[5:]  # Пропускаем первые 5 строк
                    colors = [line.strip() for line in lines if line.strip()]
                    palette_name = os.path.splitext(os.path.basename(path))[0]
                    self.palettes[palette_name] = colors

                    # Сохраняем в JSON
                    json_path = os.path.join("palettes", f"{palette_name}.json")
                    with open(json_path, 'w') as json_file:
                        json.dump(colors, json_file)

            elif path.lower().endswith('.json'):
                with open(path, 'r') as f:
                    colors = json.load(f)
                    palette_name = os.path.splitext(os.path.basename(path))[0]
                    self.palettes[palette_name] = colors

            self.update_palette_combo()
            QMessageBox.information(self, "Успех", f"Палитра '{palette_name}' загружена!")

            # Активируем кнопки, если есть изображения для конвертации
            if self.current_image_path or self.conversion_queue:
                self.convert_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить палитру:\n{str(e)}")

    def select_palette(self, name):
        if name in self.palettes:
            self.current_palette = self.palettes[name]
            if self.current_image_path or self.conversion_queue:
                self.convert_btn.setEnabled(True)

    def add_files_to_queue(self, file_paths):
        """Добавляет файлы в очередь обработки"""
        for file_path in file_paths:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.conversion_queue.append(file_path)

        if len(file_paths) > 1:
            QMessageBox.information(
                self,
                "Добавлено в очередь",
                f"Добавлено {len(file_paths)} изображений в очередь обработки. "
                f"Всего в очереди: {len(self.conversion_queue)}"
            )

    def convert_current_image(self):
        """Конвертирует текущее изображение или начинает обработку очереди"""
        if not self.current_palette:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите палитру")
            return

        if not self.current_image_path and not self.conversion_queue:
            QMessageBox.warning(self, "Ошибка", "Нет изображений для конвертации")
            return

        # Если есть очередь, начинаем обработку
        if self.conversion_queue and not self.is_processing_queue:
            self.process_next_in_queue()
        # Иначе конвертируем текущее изображение
        elif self.current_image_path:
            self.convert_image(self.current_image_path)

    def process_next_in_queue(self):
        """Обрабатывает следующий элемент в очереди"""
        if not self.conversion_queue:
            self.is_processing_queue = False
            QMessageBox.information(self, "Готово", "Все изображения обработаны")
            return

        self.is_processing_queue = True
        image_path = self.conversion_queue.pop(0)
        self.load_image(image_path)
        self.convert_image(image_path)

    def convert_image(self, image_path):
        """Запускает конвертацию изображения"""
        self.progress_bar.setValue(0)
        self.convert_btn.setEnabled(False)

        method_map = {
            0: 'euclidean',
            1: 'ciede2000_optimized',
            2: 'weighted_euclidean',
            3: 'bt2124'
        }
        method = method_map[self.method_combo.currentIndex()]
        threads = self.threads_spin.value()

        self.converter = ImageConverter(
            image_path,
            self.current_palette,
            method=method,
            block_size=1000,
            threads=threads
        )
        self.converter.progress_updated.connect(self.progress_bar.setValue)
        self.converter.result_ready.connect(self.show_result)
        self.converter.file_finished.connect(self.on_file_converted)
        self.converter.finished.connect(self.on_conversion_finished)
        self.converter.start()

    def show_result(self, img_array):
        """Отображение результата конвертации"""
        try:
            height, width, _ = img_array.shape
            self.result_image = Image.fromarray(img_array, 'RGB')

            # Конвертируем в QImage
            bytes_per_line = 3 * width
            qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Отображаем результат
            self.result_label.setPixmap(QPixmap.fromImage(qimage).scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        except Exception as e:
            print(f"Error showing result: {e}")

    def on_file_converted(self, output_path):
        """Вызывается при завершении конвертации одного файла"""
        print(f"Файл сохранен: {output_path}")

    def on_conversion_finished(self):
        """Вызывается при завершении конвертации"""
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Если есть очередь, обрабатываем следующий файл
        if self.conversion_queue and self.is_processing_queue:
            self.process_next_in_queue()
        else:
            self.is_processing_queue = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_paths = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_paths.append(file_path)
            elif file_path.lower().endswith(('.txt', '.json')):
                self.load_palette_file(file_path)

        if file_paths:
            if len(file_paths) == 1:
                self.load_image(file_paths[0])
            else:
                self.add_files_to_queue(file_paths)
                self.convert_btn.setEnabled(self.current_palette is not None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(9)
    app.setFont(font)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
