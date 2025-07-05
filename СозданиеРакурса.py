import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL
from PIL import Image, ImageTk

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Создание нового ракурса из фотографий")
        self.root.geometry("800x600")
        
        # Инициализация переменных
        self.original_img1 = None
        self.original_img2 = None
        self.aligned_img1 = None
        self.aligned_img2 = None
        self.disparity = None
        self.result_image = None
        self.tk_images = []
        
        # Создание интерфейса
        self.create_widgets()
        
        # Центрирование окна
        self.center_window()
    
    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
    
    def create_widgets(self):
        # Основной фрейм
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Заголовок
        title_label = tk.Label(
            main_frame, 
            text="Создание промежуточного ракурса из двух фотографий",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Кнопка загрузки изображений
        load_btn = tk.Button(
            main_frame, 
            text="Загрузить 2 изображения", 
            command=self.load_images, 
            height=2, 
            width=20,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10)
        )
        load_btn.pack(pady=15)
        
        # Область для превью
        self.preview_frame = tk.LabelFrame(main_frame, text="Превью")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Инструкция
        instruction = tk.Label(
            self.preview_frame,
            text="Загрузите 2 фотографии одного объекта с разных ракурсов",
            fg="gray"
        )
        instruction.pack(pady=50)
        
        # Слайдер для настройки
        self.alpha_var = tk.DoubleVar(value=0.5)
        scale_frame = tk.Frame(main_frame)
        scale_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(scale_frame, text="Смещение ракурса:").pack(side=tk.LEFT, padx=5)
        self.scale = Scale(
            scale_frame, 
            variable=self.alpha_var, 
            from_=0.0, 
            to=1.0, 
            resolution=0.05, 
            orient=HORIZONTAL, 
            length=400,
            command=self.update_preview
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scale.config(state=tk.DISABLED)
        
        # Кнопки управления
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        self.process_btn = tk.Button(
            btn_frame, 
            text="Обработать", 
            command=self.process_images, 
            state=tk.DISABLED,
            bg="#2196F3",
            fg="white",
            width=15
        )
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        self.save_btn = tk.Button(
            btn_frame, 
            text="Сохранить результат", 
            command=self.save_result, 
            state=tk.DISABLED,
            bg="#FF9800",
            fg="white",
            width=15
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)
    
    def load_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Выберите 2 фотографии с разных ракурсов",
            filetypes=[
                ("Изображения", "*.jpg;*.jpeg;*.png;*.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_paths or len(file_paths) < 2:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите минимум 2 изображения")
            return
        
        # Загружаем и масштабируем изображения
        self.original_img1 = self.load_and_scale(file_paths[0])
        self.original_img2 = self.load_and_scale(file_paths[1])
        
        if self.original_img1 is None or self.original_img2 is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображения")
            return
        
        # Показываем превью загруженных изображений
        self.show_image_previews(self.original_img1, self.original_img2)
        
        # Активируем кнопку обработки
        self.process_btn.config(state=tk.NORMAL)
        self.scale.config(state=tk.NORMAL)
    
    def load_and_scale(self, path, max_size=800):
        img = cv2.imread(path)
        if img is None:
            return None
        
        # Масштабирование для производительности
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, 
                            interpolation=cv2.INTER_AREA)
        return img
    
    def show_image_previews(self, img1, img2):
        # Очищаем предыдущие превью
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        # Создаем контейнер
        container = tk.Frame(self.preview_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Фреймы для изображений
        frame1 = tk.Frame(container)
        frame1.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        frame2 = tk.Frame(container)
        frame2.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Отображаем первое изображение
        tk_img1 = self.cv2_to_tkinter(img1)
        label1 = tk.Label(frame1, image=tk_img1)
        label1.image = tk_img1
        label1.pack()
        self.tk_images.append(tk_img1)
        
        tk.Label(frame1, text="Изображение 1", font=("Arial", 9)).pack(pady=5)
        
        # Отображаем второе изображение
        tk_img2 = self.cv2_to_tkinter(img2)
        label2 = tk.Label(frame2, image=tk_img2)
        label2.image = tk_img2
        label2.pack()
        self.tk_images.append(tk_img2)
        
        tk.Label(frame2, text="Изображение 2", font=("Arial", 9)).pack(pady=5)
    
    def cv2_to_tkinter(self, cv2_img, max_size=300):
        """Конвертация изображения OpenCV в формат Tkinter"""
        if cv2_img is None:
            return None
            
        # Конвертация цветов
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Масштабирование
        ratio = min(max_size / pil_img.width, max_size / pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        
        return ImageTk.PhotoImage(pil_img)
    
    def process_images(self):
        # Выравниваем изображения
        self.aligned_img1, self.aligned_img2 = self.align_two_images(
            self.original_img1, self.original_img2
        )
        
        # Проверяем, что выравнивание прошло успешно
        if self.aligned_img1 is None or self.aligned_img2 is None:
            messagebox.showerror("Ошибка", "Не удалось выровнить изображения")
            return
        
        # Вычисляем карту смещений
        self.disparity = self.calculate_disparity(
            self.aligned_img1, self.aligned_img2
        )
        
        # Создаем промежуточное изображение
        self.update_preview()
        
        # Активируем кнопку сохранения
        self.save_btn.config(state=tk.NORMAL)
        self.save_btn.focus_set()
    
    def align_two_images(self, img1, img2):
        # Используем ORB для совместимости
        orb = cv2.ORB_create(1000)
        
        # Находим ключевые точки
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            messagebox.showwarning("Предупреждение", "Недостаточно ключевых точек для выравнивания")
            return img1, img2
        
        # Сопоставляем точки
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            messagebox.showwarning("Предупреждение", "Недостаточно совпадений для выравнивания")
            return img1, img2
        
        # Отбираем лучшие совпадения
        matches = sorted(matches, key=lambda x: x.distance)[:30]
        
        # Получаем координаты точек
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Находим гомографию
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return img1, img2
        
        # Применяем трансформацию ко второму изображению
        aligned_img2 = cv2.warpPerspective(
            img2, M, (img1.shape[1], img1.shape[0])
        
        return img1, aligned_img2
    
    def calculate_disparity(self, img1, img2):
        # Конвертируем в оттенки серого
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Создаем стерео матчер
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=7,
            P1=8*3*7**2,
            P2=32*3*7**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Вычисляем карту смещений
        disparity = stereo.compute(gray1, gray2)
        
        # Нормализуем для визуализации
        return cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def create_intermediate_view(self, alpha=0.5):
        if self.aligned_img1 is None or self.aligned_img2 is None or self.disparity is None:
            return None
        
        # Убедимся, что изображения имеют одинаковый размер
        h, w = self.aligned_img1.shape[:2]
        if self.aligned_img2.shape[0] != h or self.aligned_img2.shape[1] != w:
            self.aligned_img2 = cv2.resize(self.aligned_img2, (w, h))
        
        if self.disparity.shape[0] != h or self.disparity.shape[1] != w:
            self.disparity = cv2.resize(self.disparity, (w, h))
        
        result = np.zeros_like(self.aligned_img1)
        
        # Создаем промежуточное изображение
        for y in range(h):
            for x in range(w):
                disp_val = self.disparity[y, x] / 16.0
                if disp_val > 1:  # Игнорируем малые смещения
                    shift = alpha * disp_val
                    new_x = int(x - shift)
                    
                    if 0 <= new_x < w:
                        result[y, new_x] = self.aligned_img1[y, x]
        
        # Заполняем пропущенные области
        mask = result.sum(axis=2) == 0
        result[mask] = (self.aligned_img1[mask] * (1 - alpha) + self.aligned_img2[mask] * alpha).astype(np.uint8)
        
        # Улучшаем резкость
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        return cv2.filter2D(result, -1, kernel)
    
    def update_preview(self, event=None):
        alpha = self.alpha_var.get()
        self.result_image = self.create_intermediate_view(alpha)
        
        if self.result_image is None:
            return
        
        # Показываем результат
        self.show_result_preview()
    
    def show_result_preview(self):
        # Очищаем предыдущие превью
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        # Создаем контейнер
        container = tk.Frame(self.preview_frame)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Отображаем результат
        tk_result = self.cv2_to_tkinter(self.result_image, 500)
        label = tk.Label(container, image=tk_result)
        label.image = tk_result
        label.pack()
        self.tk_images.append(tk_result)
        
        tk.Label(container, text="Промежуточный ракурс", font=("Arial", 10, "bold")).pack(pady=10)
        
        # Инструкция для слайдера
        tk.Label(
            container, 
            text="Используйте слайдер выше для изменения ракурса",
            fg="gray"
        ).pack()
    
    def save_result(self):
        if self.result_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить новый ракурс",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.result_image)
            messagebox.showinfo("Сохранено", f"Изображение успешно сохранено:\n{file_path}")

def main():
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()