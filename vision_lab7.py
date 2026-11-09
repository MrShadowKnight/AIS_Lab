"""
Лабораторна робота №7. Використання бібліотеки машинного зору OpenCV.

Програма вміє:
  1) знаходити обличчя та 5 ключових точок (очі, ніс, кути рота) - модель YuNet;
  2) знаходити об'єкти 80 класів COCO (людина, кіт, чашка...) - модель YOLOX;
  3) знаходити та читати QR-коди - cv2.QRCodeDetector.
Працює з фото, відеофайлом або вебкамерою.

Приклади запуску:
  python vision_lab7.py --mode faces   --input photo.jpg
  python vision_lab7.py --mode objects --input street.mp4 --output result.mp4
  python vision_lab7.py --mode all     --input 0          # вебкамера
"""
import argparse
import time
import numpy as np
import cv2

FACE_MODEL = 'models/face_detection_yunet_2023mar.onnx'
OBJ_MODEL = 'models/object_detection_yolox_2022nov.onnx'

# 80 класів набору COCO, на яких навчена YOLOX
COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


# ---------------------------------------------------------------- обличчя
class FaceDetector:
    """Детектор облич YuNet (згорткова мережа, вбудована в OpenCV)."""

    def __init__(self, conf=0.8):
        self.net = cv2.FaceDetectorYN.create(FACE_MODEL, '', (320, 320),
                                             score_threshold=conf,
                                             nms_threshold=0.3)

    def detect(self, img):
        h, w = img.shape[:2]
        self.net.setInputSize((w, h))          # мережа працює з будь-яким розміром
        _, faces = self.net.detect(img)
        result = []
        for f in (faces if faces is not None else []):
            x, y, bw, bh = (int(v) for v in f[:4])
            points = f[4:14].reshape(5, 2).astype(int)   # 5 ключових точок
            result.append({'box': (x, y, bw, bh), 'points': points,
                           'score': float(f[14])})
        return result


# ---------------------------------------------------------------- об'єкти
class ObjectDetector:
    """Детектор об'єктів YOLOX-S, запускається через модуль cv2.dnn."""

    def __init__(self, conf=0.5, nms=0.5, size=640):
        self.net = cv2.dnn.readNet(OBJ_MODEL)
        self.conf, self.nms, self.size = conf, nms, size
        # сітки центрів комірок для трьох масштабів (крок 8, 16, 32 пікселі)
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _letterbox(self, img):
        # зменшуємо зі збереженням пропорцій і доповнюємо сірим до 640x640
        r = min(self.size / img.shape[0], self.size / img.shape[1])
        resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        return canvas, r

    def detect(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas, r = self._letterbox(rgb)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        self.net.setInput(blob)
        out = self.net.forward()[0]             # 8400 x 85

        # декодуємо: центр і розмір рамки, "об'єктність", ймовірності класів
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        keep = conf > self.conf
        boxes = np.hstack((xy - wh / 2, wh))[keep] / r
        cls, conf = cls[keep], conf[keep]

        # прибираємо рамки, що дублюють одна одну (Non-Maximum Suppression)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(),
                                      cls.tolist(), self.conf, self.nms)
        return [{'box': tuple(int(v) for v in boxes[i]), 'label': COCO[cls[i]],
                 'score': float(conf[i])} for i in np.array(idx).flatten()]


# ---------------------------------------------------------------- QR-коди
class QRReader:
    def __init__(self):
        self.det = cv2.QRCodeDetector()

    def detect(self, img):
        ok, texts, corners, _ = self.det.detectAndDecodeMulti(img)
        if not ok:
            return []
        return [{'text': t, 'corners': c.astype(int)}
                for t, c in zip(texts, corners)]


# ---------------------------------------------------------------- малювання
def color_for(label):
    rng = np.random.RandomState(sum(map(ord, label)))  # сталий колір для класу
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)


def draw_faces(img, faces):
    for f in faces:
        x, y, w, h = f['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 0), 2)
        for (px, py), c in zip(f['points'], [(255, 0, 0), (0, 0, 255),
                                             (0, 255, 255), (255, 0, 255),
                                             (255, 255, 0)]):
            cv2.circle(img, (px, py), 3, c, -1)
        draw_label(img, f"face {f['score']:.2f}", x, y, (0, 220, 0))


def draw_objects(img, objs):
    for o in objs:
        x, y, w, h = o['box']
        c = color_for(o['label'])
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        draw_label(img, f"{o['label']} {o['score']:.2f}", x, y, c)


def draw_qr(img, codes):
    for q in codes:
        pts = q['corners'].reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255, 0, 255), 3)
        x, y = q['corners'][0]
        # cv2.putText не вміє кирилицю, тому текст QR друкуємо в консоль
        draw_label(img, 'QR' if q['text'] else 'QR (not decoded)', x, y,
                   (255, 0, 255))


# ---------------------------------------------------------------- обробка
class VisionApp:
    def __init__(self, mode='all'):
        self.mode = mode
        self.faces = FaceDetector() if mode in ('faces', 'all') else None
        self.objects = ObjectDetector() if mode in ('objects', 'all') else None
        self.qr = QRReader() if mode in ('qr', 'all') else None

    def process(self, img):
        """Повертає зображення з розміткою і словник результатів."""
        out, res = img.copy(), {}
        if self.objects:
            res['objects'] = self.objects.detect(img)
            draw_objects(out, res['objects'])
        if self.faces:
            res['faces'] = self.faces.detect(img)
            draw_faces(out, res['faces'])
        if self.qr:
            res['qr'] = self.qr.detect(img)
            draw_qr(out, res['qr'])
        return out, res


def print_results(res):
    for f in res.get('faces', []):
        print(f"  обличчя: рамка {f['box']}, впевненість {f['score']:.2f}")
    for o in res.get('objects', []):
        print(f"  об'єкт: {o['label']:<12} {o['score']:.2f}  рамка {o['box']}")
    for q in res.get('qr', []):
        print(f"  QR-код: \"{q['text']}\"" if q['text'] else
              '  QR-код знайдено, але не прочитано')
    if not any(res.values()):
        print('  нічого не знайдено')


def run_image(app, path, output=None):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f'Не вдалося відкрити {path}')
    t = time.time()
    out, res = app.process(img)
    print(f'{path}: {(time.time() - t) * 1000:.0f} мс')
    print_results(res)
    cv2.imwrite(output or 'result.jpg', out)
    return out, res


def run_video(app, source, output=None, show=True):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        out, res = app.process(frame)
        fps = 1 / max(time.time() - t, 1e-6)
        cv2.putText(out, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        if output:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'),
                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(out)
        if show:
            cv2.imshow('Lab 7 - OpenCV', out)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):   # Esc або q - вихід
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Розпізнавання з OpenCV')
    p.add_argument('--mode', default='all',
                   choices=['faces', 'objects', 'qr', 'all'])
    p.add_argument('--input', required=True,
                   help='фото, відео або номер камери (0)')
    p.add_argument('--output', help='куди зберегти результат')
    a = p.parse_args()
    app = VisionApp(a.mode)
    if a.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        run_image(app, a.input, a.output)
    else:
        run_video(app, a.input, a.output)
