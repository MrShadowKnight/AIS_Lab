ЛР-7. Розпізнавання з OpenCV (Пінчак В.Б., КН-2327Б)

Встановлення:   pip install opencv-python numpy
(для experiments.py ще matplotlib, для make_video.py - pillow і ffmpeg)

Запуск з цієї папки:
  python vision_lab7.py --mode all     --input images/group.jpg
  python vision_lab7.py --mode objects --input images/coffee.png
  python vision_lab7.py --mode qr      --input images/qr_scene.jpg
  python vision_lab7.py --mode faces   --input 0        (вебкамера, вихід - Esc або q)

Потрібен OpenCV 4.5.4 або новіший (для cv2.FaceDetectorYN).
Результат для фото зберігається в result.jpg (або у файл з --output).
