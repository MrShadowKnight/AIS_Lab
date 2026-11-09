"""Контрольні приклади до ЛР-7: коли детектор працює, а коли ні."""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vision_lab7 import (FaceDetector, ObjectDetector, VisionApp,
                         draw_faces, draw_objects)

fd = FaceDetector()
img = cv2.imread('images/astronaut.png')


def rotate(im, angle):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(im, M, (w, h))


def best(faces):
    return max([f['score'] for f in faces], default=0)


def tile(im, title, ok, size=220):
    """Мініатюра з підписом: зелена рамка - знайдено, червона - ні."""
    im = cv2.resize(im, (size, int(im.shape[0] * size / im.shape[1])))
    im = cv2.copyMakeBorder(im, 4, 4, 4, 4, cv2.BORDER_CONSTANT,
                            value=(0, 170, 0) if ok else (0, 0, 220))
    bar = np.full((30, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack((im, bar))


def strip(tiles, cols):
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    h = max(t.shape[0] for t in tiles)
    pad = lambda t: cv2.copyMakeBorder(t, 0, h - t.shape[0], 5, 5,
                                       cv2.BORDER_CONSTANT, value=(255,) * 3)
    lines = [np.hstack([pad(t) for t in r] +
                       [np.full((h, tiles[0].shape[1] + 10, 3), 255, np.uint8)]
                       * (cols - len(r))) for r in rows]
    return np.vstack([cv2.copyMakeBorder(l, 5, 5, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(255,) * 3) for l in lines])


# 1. Поворот обличчя від 0 до 180 градусів
angles = list(range(0, 181, 15))
scores, tiles = [], []
for a in angles:
    im = rotate(img, a)
    faces = fd.detect(im)
    scores.append(best(faces))
    if a in (0, 30, 60, 90, 105, 135, 150, 180):
        out = im.copy()
        draw_faces(out, faces)
        tiles.append(tile(out, f'{a} deg: ' + (f'{best(faces):.2f}' if faces
                                               else 'not found'), bool(faces)))
cv2.imwrite('results/exp_rotation.jpg', strip(tiles, 4))
print('Поворот:', dict(zip(angles, np.round(scores, 2))))

plt.figure(figsize=(8, 3.6))
plt.plot(angles, scores, 'o-', color='tab:green')
plt.axhline(0.8, ls='--', color='gray', label='поріг 0.8')
plt.axvspan(0, 30, color='tab:blue', alpha=0.12,
            label='межа Хаара за методичкою (~30°)')
plt.xlabel('Кут повороту, °'); plt.ylabel('Впевненість (0 – не знайдено)')
plt.ylim(-0.05, 1.0); plt.xticks(angles); plt.grid(alpha=0.3); plt.legend()
plt.title('Впевненість детектора облич залежно від кута повороту')
plt.tight_layout(); plt.savefig('results/exp_rotation_chart.png', dpi=120)

# 2. Яскравість і розмір обличчя
tiles = []
for k in (1, 0.2, 0.05, 0.03):
    im = (img * k).astype(np.uint8)
    faces = fd.detect(im); out = im.copy(); draw_faces(out, faces)
    tiles.append(tile(out, f'brightness {int(k * 100)}%: ' +
                      (f'{best(faces):.2f}' if faces else 'not found'),
                      bool(faces)))
for s in (0.3, 0.2, 0.15, 0.1):
    im = cv2.resize(img, None, fx=s, fy=s)
    faces = fd.detect(im); out = im.copy(); draw_faces(out, faces)
    big = cv2.resize(out, (512, 512), interpolation=cv2.INTER_NEAREST)
    tiles.append(tile(big, f'face {int(113 * s)} px: ' +
                      (f'{best(faces):.2f}' if faces else 'not found'),
                      bool(faces)))
cv2.imwrite('results/exp_dark_scale.jpg', strip(tiles, 4))

# 3. Частково закрите обличчя
tiles = []
for name, (x0, y0, x1, y1) in {'eyes covered': (178, 95, 267, 125),
                               'mouth covered': (178, 140, 267, 175),
                               '2/3 covered': (178, 62, 267, 140)}.items():
    im = img.copy()
    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 0), -1)
    faces = fd.detect(im); out = im.copy(); draw_faces(out, faces)
    crop = out[20:260, 110:340]
    tiles.append(tile(crop, f'{name}: ' + (f'{best(faces):.2f}' if faces
                                           else 'not found'), bool(faces)))
cv2.imwrite('results/exp_occlusion.jpg', strip(tiles, 3))

# 4. Об'єкти, яких немає серед 80 класів COCO
od_low = ObjectDetector(conf=0.25)
tiles = []
for n in ('rocket.png', 'china.jpg'):
    im = cv2.imread('images/' + n)
    for det, thr in ((ObjectDetector(), 0.5), (od_low, 0.25)):
        objs = det.detect(im); out = im.copy(); draw_objects(out, objs)
        txt = ', '.join(f"{o['label']} {o['score']:.2f}" for o in objs) or 'none'
        tiles.append(tile(out, f'conf>{thr}: {txt}', bool(objs), size=300))
        print(n, thr, txt)
cv2.imwrite('results/exp_unknown.jpg', strip(tiles, 2))

# 5. Погані QR-коди
app = VisionApp('qr')
tiles = []
for n, t in (('qr_damaged.jpg', 'damaged'), ('qr_small.jpg', '160x107 px')):
    im = cv2.imread('images/' + n)
    out, res = app.process(im)
    big = cv2.resize(out, (480, 320), interpolation=cv2.INTER_NEAREST)
    tiles.append(tile(big, f"{t}: read {len(res['qr'])} of 3", False, 420))
    print(n, [q['text'] for q in res['qr']])
cv2.imwrite('results/exp_qr_bad.jpg', strip(tiles, 2))
