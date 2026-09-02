from PIL import Image


def generate_mirage(up_img: Image.Image, hide_img: Image.Image) -> Image.Image:
    """把两张图合成幻影坦克。"""
    max_w = max(up_img.size[0], hide_img.size[0])
    up_img = up_img.resize((max_w, int(up_img.size[1] * (max_w / up_img.size[0]))))
    hide_img = hide_img.resize((max_w, int(hide_img.size[1] * (max_w / hide_img.size[0]))))
    max_size = (max_w, max(up_img.size[1], hide_img.size[1]))

    if hide_img.size[1] == up_img.size[1]:
        up_img = up_img.convert("L")
        hide_img = hide_img.convert("L")
    elif max_size[1] == hide_img.size[1]:
        up_img_temp = Image.new("RGBA", max_size, (255, 255, 255, 255))
        up_img_temp.paste(up_img, (0, (max_size[1] - up_img.size[1]) // 2))
        up_img = up_img_temp.convert("L")
        hide_img = hide_img.convert("L")
    else:
        hide_img_temp = Image.new("RGBA", max_size, (0, 0, 0, 255))
        hide_img_temp.paste(hide_img, (0, (max_size[1] - hide_img.size[1]) // 2))
        up_img = up_img.convert("L")
        hide_img = hide_img_temp.convert("L")

    up = list(up_img.getdata())
    hide = list(hide_img.getdata())
    pixels = []
    for la_raw, lb_raw in zip(up, hide):
        la = (la_raw / 512) + 0.5
        lb = lb_raw / 512
        denom = 1 - (la - lb)
        if denom <= 1e-6:
            denom = 1e-6
        r = int((255 * lb) / denom)
        a = int(denom * 255)
        r = max(0, min(255, r))
        a = max(0, min(255, a))
        pixels.append((r, r, r, a))
    out = Image.new("RGBA", max_size, (255, 255, 255, 255))
    out.putdata(pixels)
    return out
