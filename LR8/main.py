import cv2
import sys
import argparse


# Функция обнаружения лиц (без изменений)
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


# Обработка изображения по пути к файлу
def process_image(file_path, net):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {file_path}")
        return None
    result_img, _ = highlightFace(net, img)
    return result_img


# Обработка видеопотока с камеры
def process_camera(net):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Ошибка: камера недоступна")
        return

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            print("Ошибка: не удалось получить кадр")
            break

        result_img, _ = highlightFace(net, frame)
        cv2.imshow("Face Detection - Camera", result_img)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # Выход по ESC или 'q'
            break

    video.release()
    cv2.destroyAllWindows()


# Основная функция CLI
def main_cli():
    parser = argparse.ArgumentParser(description='Face Detection Tool')
    parser.add_argument('-f', '--file', type=str, help='Path to image file')
    args = parser.parse_args()

    # Загрузка модели
    face_net = cv2.dnn.readNet(
        "opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt"
    )

    if args.file:
        # Режим обработки файла
        result = process_image(args.file, face_net)
        if result is not None:
            cv2.imshow("Face Detection - Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # Режим камеры по умолчанию
        process_camera(face_net)


# Запуск GUI версии
def main_gui():
    import tkinter as tk
    from tkinter import filedialog
    from PIL import Image, ImageTk

    # Загрузка модели при запуске GUI
    face_net = cv2.dnn.readNet(
        "opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt"
    )

    def open_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            process_and_show_image(file_path)

    def start_camera():
        root.destroy()
        process_camera(face_net)

    def process_and_show_image(file_path):
        result_img = process_image(file_path, face_net)
        if result_img is not None:
            # Конвертация для Tkinter
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(result_img)
            img_tk = ImageTk.PhotoImage(img_pil)

            # Обновление интерфейса
            img_label.config(image=img_tk)
            img_label.image = img_tk
            status_label.config(text=f"Обработано: {file_path}")

    # Создание GUI
    root = tk.Tk()
    root.title("Face Detector GUI")
    root.geometry("800x600")

    # Кнопки управления
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    btn_open = tk.Button(btn_frame, text="Выбрать изображение", command=open_image)
    btn_open.pack(side=tk.LEFT, padx=5)

    btn_camera = tk.Button(btn_frame, text="Запустить камеру", command=start_camera)
    btn_camera.pack(side=tk.LEFT, padx=5)

    # Область для изображения
    img_label = tk.Label(root)
    img_label.pack(pady=10, fill=tk.BOTH, expand=True)

    # Статус бар
    status_label = tk.Label(root, text="Выберите действие", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()


# Определение точки входа
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()  # CLI режим при наличии аргументов
    else:
        main_gui()  # GUI режим по умолчанию