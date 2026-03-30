#!/usr/bin/env python3
"""生成 IHC Score Analyzer 应用图标"""

from PIL import Image, ImageDraw, ImageFont
import math
import os

SIZE = 1024
CENTER = SIZE // 2
BG_COLOR = (30, 30, 30)


def draw_icon():
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── 圆角矩形背景 ──
    r = 180  # 圆角半径
    draw.rounded_rectangle([40, 40, SIZE - 40, SIZE - 40],
                           radius=r, fill=(24, 24, 32, 255))

    # ── 外圈光环 ──
    draw.rounded_rectangle([40, 40, SIZE - 40, SIZE - 40],
                           radius=r, outline=(79, 195, 247, 180), width=6)

    # ── 模拟显微镜下的细胞切片（圆形区域）──
    circle_cx, circle_cy = CENTER, CENTER - 40
    circle_r = 320

    # 圆形裁剪区域背景（浅粉色，模拟 H&E 染色背景）
    draw.ellipse([circle_cx - circle_r, circle_cy - circle_r,
                  circle_cx + circle_r, circle_cy + circle_r],
                 fill=(245, 228, 220, 255),
                 outline=(100, 100, 120, 200), width=4)

    # ── 画"细胞"散点，模拟 IHC 染色 ──
    import random
    random.seed(42)

    cells = []
    for _ in range(200):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, circle_r - 30)
        cx = int(circle_cx + dist * math.cos(angle))
        cy = int(circle_cy + dist * math.sin(angle))
        cr = random.randint(8, 22)
        cells.append((cx, cy, cr))

    # 阴性细胞（蓝紫色，H&E 风格）
    for cx, cy, cr in cells[:100]:
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                     fill=(130, 120, 180, 200))
        # 细胞核
        nr = cr // 2
        draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr],
                     fill=(80, 60, 140, 230))

    # 弱阳性（浅棕色）
    for cx, cy, cr in cells[100:130]:
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                     fill=(200, 170, 120, 220))
        nr = cr // 2
        draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr],
                     fill=(170, 130, 80, 240))

    # 阳性（棕色）
    for cx, cy, cr in cells[130:165]:
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                     fill=(180, 120, 60, 230))
        nr = cr // 2
        draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr],
                     fill=(140, 80, 30, 250))

    # 强阳性（深棕色）
    for cx, cy, cr in cells[165:200]:
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                     fill=(140, 70, 20, 240))
        nr = cr // 2
        draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr],
                     fill=(100, 40, 10, 255))

    # ── 底部评分条（四色横条）──
    bar_y = SIZE - 200
    bar_h = 40
    bar_left = 120
    bar_right = SIZE - 120
    bar_w = bar_right - bar_left
    colors = [
        ((66, 165, 245), 0.25),    # 阴性 - 蓝
        ((102, 187, 106), 0.20),   # 弱阳性 - 绿
        ((255, 167, 38), 0.25),    # 阳性 - 橙
        ((239, 83, 80), 0.30),     # 强阳性 - 红
    ]
    x = bar_left
    for color, ratio in colors:
        w = int(bar_w * ratio)
        draw.rounded_rectangle([x, bar_y, x + w, bar_y + bar_h],
                               radius=6, fill=color + (230,))
        x += w

    # ── "IHC" 文字 ──
    try:
        font_large = ImageFont.truetype("Times New Roman", 130)
    except OSError:
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 130)
        except OSError:
            font_large = ImageFont.load_default()

    text = "IHC"
    bbox = draw.textbbox((0, 0), text, font=font_large)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (SIZE - tw) // 2
    ty = bar_y + bar_h + 15

    # 文字阴影
    draw.text((tx + 3, ty + 3), text, fill=(0, 0, 0, 150), font=font_large)
    # 文字主体
    draw.text((tx, ty), text, fill=(79, 195, 247, 255), font=font_large)

    return img


def main():
    icon = draw_icon()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(out_dir)

    # 保存 PNG
    png_path = os.path.join(root_dir, "icon.png")
    icon.save(png_path, "PNG")
    print(f"PNG: {png_path}")

    # 生成 ICO (Windows, 多尺寸)
    ico_path = os.path.join(root_dir, "icon.ico")
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon_resized = [icon.resize(s, Image.LANCZOS) for s in sizes]
    icon_resized[0].save(ico_path, format='ICO', sizes=sizes, append_images=icon_resized[1:])
    print(f"ICO: {ico_path}")

    # 生成 ICNS (macOS)
    icns_path = os.path.join(root_dir, "icon.icns")
    iconset_dir = os.path.join(root_dir, "icon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    icns_sizes = {
        'icon_16x16.png': 16, 'icon_16x16@2x.png': 32,
        'icon_32x32.png': 32, 'icon_32x32@2x.png': 64,
        'icon_128x128.png': 128, 'icon_128x128@2x.png': 256,
        'icon_256x256.png': 256, 'icon_256x256@2x.png': 512,
        'icon_512x512.png': 512, 'icon_512x512@2x.png': 1024,
    }
    for name, sz in icns_sizes.items():
        icon.resize((sz, sz), Image.LANCZOS).save(
            os.path.join(iconset_dir, name), "PNG")

    os.system(f"iconutil -c icns {iconset_dir} -o {icns_path}")
    # 清理 iconset
    import shutil
    shutil.rmtree(iconset_dir, ignore_errors=True)
    print(f"ICNS: {icns_path}")


if __name__ == "__main__":
    main()
