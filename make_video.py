"""Демо-відео ЛР-7: програма покадрово обробляє відеопотік з віртуальної камери."""
import subprocess, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from vision_lab7 import VisionApp, FaceDetector

W, H, FPS = 1280, 720, 25
VW, VH = 820, 615                     # "вікно камери"
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
f = lambda p, s: ImageFont.truetype(p, s)
BG, PANEL, ACC = (24, 26, 32), (36, 39, 48), (90, 200, 120)

ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                       '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                       '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                       '-crf', '22', '-pix_fmt', 'yuv420p', 'lab7_demo.mp4'],
                      stdin=subprocess.PIPE)
n_frames = [0]


def emit(pil):
    ff.stdin.write(np.asarray(pil.convert('RGB')).tobytes())
    n_frames[0] += 1


def ease(t):
    return t * t * (3 - 2 * t)


def camera(img, cx, cy, zoom, angle=0.0, bright=1.0):
    """Віртуальна камера: центр (cx, cy) у частках, zoom, поворот, яскравість."""
    h, w = img.shape[:2]
    s = min(VW / w, VH / h) * zoom
    M = cv2.getRotationMatrix2D((cx * w, cy * h), angle, s)
    M[0, 2] += VW / 2 - cx * w
    M[1, 2] += VH / 2 - cy * h
    fr = cv2.warpAffine(img, M, (VW, VH), borderValue=(20, 20, 20))
    return (fr * bright).astype(np.uint8) if bright != 1 else fr


def layout(section, step, frame_bgr, lines, caption, stats):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 52], fill=PANEL)
    d.text((20, 12), 'Lab 7 — OpenCV vision app', font=f(BOLD, 22), fill=ACC)
    d.text((370, 14), f'{step}  ·  {section}', font=f(FONT, 20), fill=(230, 230, 230))
    im.paste(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), (20, 66))
    x0 = VW + 40
    d.rectangle([x0, 66, W - 20, 66 + VH], fill=PANEL)
    d.text((x0 + 16, 80), 'Результати', font=f(BOLD, 20), fill=(255, 255, 255))
    y = 116
    for txt, col in lines[:14]:
        d.text((x0 + 16, y), txt, font=f(MONO, 18), fill=col)
        y += 30
    d.text((x0 + 16, 66 + VH - 64), stats, font=f(MONO, 17), fill=(180, 180, 180))
    d.rectangle([0, H - 38, W, H], fill=PANEL)
    d.text((20, H - 31), caption, font=f(FONT, 18), fill=(235, 235, 235))
    return im


def result_lines(res):
    L = []
    for fc in res.get('faces', []):
        L.append((f"обличчя      {fc['score']:.2f}", (110, 230, 110)))
    for o in res.get('objects', []):
        L.append((f"{o['label']:<12} {o['score']:.2f}", (130, 190, 255)))
    for q in res.get('qr', []):
        L.append((f"QR: {q['text'][:24]}", (255, 120, 255)))
    return L or [('нічого не знайдено', (255, 120, 120))]


def run_segment(app, img, path, seconds, section, step, caption, every=1, face_px=False):
    """path(t) -> (cx, cy, zoom, angle, bright); обробка кожного кадру."""
    n = int(seconds * FPS)
    last = None
    for i in range(n):
        cx, cy, z, a, b = path(i / max(n - 1, 1))
        fr = camera(img, cx, cy, z, a, b)
        if i % every == 0 or last is None:
            t = time.time()
            out, res = app.process(fr)
            ms = (time.time() - t) * 1000
            last = (res, ms)
        else:
            out = fr.copy()
            from vision_lab7 import draw_faces, draw_objects, draw_qr
            res, ms = last
            draw_objects(out, res.get('objects', []))
            draw_faces(out, res.get('faces', []))
            draw_qr(out, res.get('qr', []))
        res, ms = last
        if a:
            extra = f'кут {a:.0f}°'
        elif b < 1:
            extra = f'яскравість {b*100:.0f}%'
        elif face_px:
            extra = f'обличчя ≈ {113 * 1.2 * z:.0f} px'
        else:
            extra = f'масштаб x{z:.2f}'
        cv2.putText(out, f'{1000/ms:.1f} FPS  {ms:.0f} ms', (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        emit(layout(section, step, out, result_lines(res), caption,
                    f'кадр {i+1}/{n}\n{extra}'))


def title(lines, seconds):
    for i in range(int(seconds * FPS)):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        y = 200
        for txt, size, col in lines:
            fnt = f(BOLD if size > 30 else FONT, size)
            tw = d.textlength(txt, font=fnt)
            d.text(((W - tw) / 2, y), txt, font=fnt, fill=col)
            y += size + 22
        emit(im)


def terminal(cmd, out_lines, result_img, seconds):
    n = int(seconds * FPS)
    res = cv2.imread(result_img)
    for i in range(n):
        t = i / n
        im = Image.new('RGB', (W, H), (12, 12, 16))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 34], fill=(50, 50, 58))
        d.text((16, 7), 'Terminal — python vision_lab7.py', font=f(FONT, 16), fill=(200, 200, 200))
        typed = cmd[:int(len(cmd) * min(t / 0.3, 1))]
        d.text((20, 50), '$ ' + typed, font=f(MONO, 18), fill=(120, 230, 120))
        shown = int(len(out_lines) * max(0, min((t - 0.35) / 0.25, 1)))
        for k, line in enumerate(out_lines[:shown]):
            d.text((20, 84 + k * 26), line, font=f(MONO, 17), fill=(225, 225, 225))
        if t > 0.62:
            r = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            r.thumbnail((W - 40, 360))
            im.paste(r, ((W - r.width) // 2, H - r.height - 20))
        emit(im)


# ------------------------------------------------------------ сценарій
img = {n: cv2.imread('images/' + n) for n in
       ['group.jpg', 'coffee.png', 'cat.png', 'qr_scene.jpg', 'astronaut.png',
        'rocket.png', 'china.jpg', 'grace_hopper.jpg']}
app_all, app_obj, app_qr = VisionApp('all'), VisionApp('objects'), VisionApp('qr')
app_face = VisionApp('faces')

title([('Лабораторна робота №7', 44, (255, 255, 255)),
       ('Використання бібліотеки машинного зору OpenCV', 26, ACC),
       ('Обличчя (YuNet) · об\'єкти COCO (YOLOX) · QR-коди', 22, (200, 200, 200)),
       ('Пінчак Володимир, КН-2327Б', 22, (200, 200, 200))], 5)

log = open('results/cli_log.txt', encoding='utf-8').read().split('\n\n')[0].split('\n')
terminal(log[0][2:], log[1:], 'results/cli_group.jpg', 10)

run_segment(app_all, img['group.jpg'],
            lambda t: (0.18 + 0.64 * ease(t), 0.5, 2.0 - 0.9 * np.sin(np.pi * t) * 0.6, 0, 1),
            14, 'обличчя + об\'єкти', '1/5',
            'Камера рухається по колажу: YuNet знаходить обличчя і 5 точок, YOLOX — людей і краватку')
run_segment(app_obj, img['coffee.png'],
            lambda t: (0.5, 0.5, 0.9 + 0.2 * ease(t), 0, 1), 6, 'об\'єкти COCO', '2/5',
            'Чашка, ложка і стіл — класи набору COCO (cup, spoon, dining table)')
run_segment(app_obj, img['cat.png'],
            lambda t: (0.45 + 0.1 * t, 0.5, 1.4 - 0.4 * ease(t), 0, 1), 6, 'об\'єкти COCO', '2/5',
            'Кіт — теж клас COCO, тому рамка з підписом cat')
run_segment(app_qr, img['qr_scene.jpg'],
            lambda t: (0.5 + 0.08 * np.sin(2 * np.pi * t), 0.5, 1.0 + 0.15 * np.sin(np.pi * t),
                       8 * np.sin(2 * np.pi * t), 1), 10, 'QR-коди', '3/5',
            'QR-коди з кирилицею читаються, але в русі нестабільно: найщільніший код — лише в окремих кадрах')
run_segment(app_face, img['astronaut.png'],
            lambda t: (0.43, 0.33, 2.2, 180 * t, 1), 14, 'поворот обличчя 0→180°', '4/5',
            'Тест повороту: YuNet тримає обличчя далеко за межею 30°, але на деяких кутах губить')
run_segment(app_face, img['astronaut.png'],
            lambda t: (0.43, 0.33, 2.2, 0, 1 - 0.98 * ease(t)), 7, 'затемнення', '4/5',
            'Затемнення: обличчя знаходиться ще при ~7% яскравості, при 3% — вже ні')
run_segment(app_face, img['astronaut.png'],
            lambda t: (0.43, 0.33, 2.2 * (1 - 0.975 * ease(t)), 0, 1), 8, 'віддалення', '4/5',
            'Віддалення: коли обличчя стає меншим за ~16–18 пікселів, детектор його втрачає', face_px=True)
run_segment(app_obj, img['rocket.png'],
            lambda t: (0.5, 0.5, 1.0 + 0.3 * t, 0, 1), 5, 'невідомий клас', '5/5',
            'Ракети немає серед класів COCO — модель нічого не знаходить')
run_segment(app_obj, img['china.jpg'],
            lambda t: (0.45 + 0.1 * t, 0.5, 1.1, 0, 1), 5, 'невідомий клас', '5/5',
            'Те саме з храмом: модель розпізнає лише те, чого її навчили')

title([('Підсумок', 40, (255, 255, 255)),
       ('Обличчя і люди знаходяться з впевненістю 0.85–0.95, предмети COCO — 0.5–0.9', 22, (220, 220, 220)),
       ('Не працює: дрібні обличчя (< ~16 px), яскравість 3% і нижче,', 22, (255, 160, 160)),
       ('об\'єкти поза 80 класами COCO, дрібні або пошкоджені QR-коди', 22, (255, 160, 160))], 6)

ff.stdin.close(); ff.wait()
print('frames', n_frames[0], 'seconds', n_frames[0] / FPS)
