from enum import Enum
from typing import List, Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.utils import *

from .cpp import cutout_image, shrink_image
from .mirage import generate_mirage

config = Config("imgtool")
logger = get_logger("imgtool")


class ImageType(Enum):
    Any = 1
    Animated = 2
    Static = 3
    Multiple = 4

    def __str__(self):
        return {
            ImageType.Any: "任意单图",
            ImageType.Animated: "动图",
            ImageType.Static: "静态图",
            ImageType.Multiple: "多张图片",
        }[self]

    def check_img(self, img) -> bool:
        if self == ImageType.Multiple:
            if not isinstance(img, list):
                return False
            for i in img:
                if not isinstance(i, Image.Image) or is_animated(i):
                    return False
            return True
        if self == ImageType.Any:
            return True
        if self == ImageType.Animated:
            return is_animated(img)
        return not is_animated(img)

    def check_type(self, tar) -> bool:
        if self == ImageType.Multiple:
            return self == tar
        if self == ImageType.Any or tar == ImageType.Any:
            return True
        return self == tar

    @classmethod
    def get_type(cls, img) -> "ImageType":
        if isinstance(img, list):
            return ImageType.Multiple
        if is_animated(img):
            return ImageType.Animated
        return ImageType.Static


class ImageOperation:
    all_ops = {}

    def __init__(self, name: str, input_type: ImageType, output_type: ImageType, process_type: str = "batch"):
        self.name = name
        self.input_type = input_type
        self.output_type = output_type
        self.process_type = process_type
        self.help = ""
        self.input_limit = parse_cfg_num(config.get("input_res_limit.default"))
        ImageOperation.all_ops[name] = self
        assert_and_reply(process_type in ["single", "batch"], f"图片操作类型{process_type}错误")
        assert_and_reply(not (input_type == ImageType.Multiple and process_type == "batch"), "多张图片操作不能以批量方式处理")

    def parse_args(self, args: List[str]) -> dict | None:
        return None

    def operate(self, img, args=None, image_type=None, frame_idx: int = 0, total_frame: int = 1):
        raise NotImplementedError()

    def __call__(self, img, args: List[str]):
        try:
            parsed = self.parse_args(args)
        except Exception as e:
            msg = f"参数错误: {e}\n{self.help}" if str(e) else f"参数错误\n{self.help}"
            raise ReplyException(msg.strip()) from e

        def apply_limit(cur):
            if isinstance(cur, Image.Image) and not is_animated(cur):
                w, h = cur.size
                cur = limit_image_by_pixels(cur, self.input_limit)
                if (w, h) != cur.size:
                    logger.info("图片操作 %s 对超限输入进行缩放 %sx%s -> %sx%s", self.name, w, h, *cur.size)
                return cur
            is_single_gif = False
            duration = 50
            if isinstance(cur, Image.Image):
                is_single_gif = True
                duration = get_gif_duration(cur)
                cur = gif_to_frames(cur)
            w, h, n = cur[0].size[0], cur[0].size[1], len(cur)
            cur = limit_image_by_pixels(cur, self.input_limit)
            new_w, new_h, new_n = cur[0].size[0], cur[0].size[1], len(cur)
            if (n, w, h) != (new_n, new_w, new_h):
                logger.info("图片操作 %s 对超限输入进行缩放 %sx%sx%s -> %sx%sx%s", self.name, n, w, h, new_n, new_w, new_h)
            if is_single_gif:
                cur = frames_to_gif(cur, int(duration * new_n / max(1, n)))
            return cur

        def process_image(cur):
            img_type = ImageType.get_type(cur)
            if self.process_type == "single":
                return self.operate(apply_limit(cur), parsed)
            if img_type == ImageType.Animated:
                frames = gif_to_frames(cur)
                frames = apply_limit(frames)
                frames = [self.operate(f, parsed, img_type, i, cur.n_frames) for i, f in enumerate(frames)]
                return frames_to_gif(frames, get_gif_duration(cur))
            return self.operate(apply_limit(cur), parsed, img_type)

        img_type = ImageType.get_type(img)
        logger.info("执行图片操作:%s 输入类型:%s 参数:%s", self.name, img_type, parsed)
        if self.input_type != ImageType.Multiple and img_type == ImageType.Multiple:
            logger.info("为 %s 操作批量处理 %s 张图片", self.name, len(img))
            return [process_image(i) for i in img]
        return process_image(img)


class GifOperation(ImageOperation):
    def __init__(self):
        super().__init__("gif", ImageType.Static, ImageType.Static, "single")
        self.help = """
将静态PNG图片转换为GIF，让透明部分能够在聊天中正确显示，使用方式:
gif n 使用普通算法生成GIF
gif 使用优化算法以默认50%不透明度阈值生成GIF
gif 0.8 使用优化算法以80%不透明度阈值生成GIF
""".strip()

    def parse_args(self, args: List[str]) -> dict:
        ret = {"opt": True, "threshold": 0.5}
        if args:
            if "n" in args:
                ret["opt"] = False
            else:
                ret["threshold"] = float(args[0])
                assert_and_reply(0.0 <= ret["threshold"] <= 1.0, "不透明度阈值必须在0-1之间")
        return ret

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        if is_animated(img):
            return img
        with TempFilePath("gif") as tmp_path:
            if args["opt"]:
                save_transparent_static_gif(img, tmp_path, args["threshold"])
            else:
                img.convert("RGBA").save(tmp_path, save_all=True, append_images=[], duration=0, loop=0)
            return open_image(tmp_path)


class PngOperation(ImageOperation):
    def __init__(self):
        super().__init__("png", ImageType.Static, ImageType.Static, "batch")
        self.help = "将图片转换为png格式"

    def parse_args(self, args: List[str]):
        assert_and_reply(not args, "该操作不接受参数")
        return None

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        return img.convert("RGBA")


class ResizeOperation(ImageOperation):
    def __init__(self):
        super().__init__("resize", ImageType.Any, ImageType.Any, "batch")
        self.help = """
缩放图像，使用方式:
resize 256 128: 缩放到256x128
resize 256: 保持宽高比缩放到长边为256
resize 0.5x: 保持宽高比缩放到原图50%
resize 3.0x 2.0x: 宽缩放3倍高缩放2倍
""".strip()

    def parse_args(self, args: List[str]) -> dict:
        ret = {"w_scale": None, "h_scale": None, "w": None, "h": None, "max": None}
        if len(args) == 1:
            if args[0].endswith("x"):
                ret["w_scale"] = float(args[0].removesuffix("x"))
                ret["h_scale"] = float(args[0].removesuffix("x"))
            else:
                ret["max"] = int(args[0])
        elif len(args) == 2:
            ret["w_scale" if args[0].endswith("x") else "w"] = float(args[0].removesuffix("x")) if args[0].endswith("x") else int(args[0])
            ret["h_scale" if args[1].endswith("x") else "h"] = float(args[1].removesuffix("x")) if args[1].endswith("x") else int(args[1])
        else:
            raise Exception()
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        w, h = img.size
        if args["max"] is not None:
            if w > h:
                h = int(args["max"] * h / w)
                w = args["max"]
            else:
                w = int(args["max"] * w / h)
                h = args["max"]
        else:
            if args["w_scale"] is not None:
                w = int(w * args["w_scale"])
            if args["h_scale"] is not None:
                h = int(h * args["h_scale"])
            if args["w"] is not None:
                w = args["w"]
            if args["h"] is not None:
                h = args["h"]
        assert_and_reply(0 < w * h * total_frame <= 1024 * 1024 * 16, f"图片尺寸{w}x{h}超出限制")
        return img.resize((w, h), Image.Resampling.BILINEAR)


class MirrorOperation(ImageOperation):
    def __init__(self):
        super().__init__("mirror", ImageType.Any, ImageType.Any, "batch")
        self.help = "镜像翻转，使用方式:\nmirror: 水平镜像\nmirror v: 垂直镜像"

    def parse_args(self, args: List[str]) -> dict:
        args = [arg[0].lower() for arg in args]
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        return {"mode": "v" if "v" in args else "h"}

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return img.transpose(Image.FLIP_TOP_BOTTOM if args["mode"] == "v" else Image.FLIP_LEFT_RIGHT)


class RotateOperation(ImageOperation):
    def __init__(self):
        super().__init__("rotate", ImageType.Any, ImageType.Any, "batch")
        self.help = "旋转图像，使用方式:\nrotate 90: 逆时针旋转90度"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个角度参数")
        return {"degree": int(args[0])}

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return img.rotate(args["degree"], expand=True)


class BackOperation(ImageOperation):
    def __init__(self):
        super().__init__("back", ImageType.Animated, ImageType.Animated, "single")
        self.help = "将动图在时间上反向播放"

    def parse_args(self, args: List[str]):
        assert_and_reply(not args, "该操作不接受参数")
        return None

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        frames = gif_to_frames(img)
        frames.reverse()
        return frames_to_gif(frames, get_gif_duration(img))


class SpeedOperation(ImageOperation):
    def __init__(self):
        super().__init__("speed", ImageType.Animated, ImageType.Animated, "single")
        self.help = """
调整动图播放速度，使用方式:
speed 2.0x 设置动图播放速度为原图的2倍
speed -2.0x 设置动图播放速度为原图的2倍倒放
speed 100 设置动图帧间隔为100ms
""".strip()

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个速度参数")
        ret = {}
        if args[0].endswith("x"):
            ret["speed"] = float(args[0].removesuffix("x"))
            if ret["speed"] < 0:
                ret["back"] = True
                ret["speed"] = -ret["speed"]
            assert_and_reply(0.01 <= ret["speed"] <= 100.0, "加速倍率必须在0.01-100.0之间")
        else:
            ret["duration"] = int(args[0])
            if ret["duration"] < 0:
                ret["back"] = True
                ret["duration"] = -ret["duration"]
            assert_and_reply(1 <= ret["duration"] <= 1000, "帧间隔必须在1ms-1000ms之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        duration = img.info.get("duration") or 100
        if "speed" in args:
            duration = duration / args["speed"]
        elif "duration" in args:
            duration = int(args["duration"])
        interval = 1
        for i in range(1, 1000):
            interval = i
            if int(duration * interval) >= 20:
                duration = int(duration * interval)
                break
        frame_num = img.n_frames
        if frame_num / interval <= 1:
            max_rate = img.info.get("duration", 100) / (20 / max(1, frame_num - 1))
            raise ReplyException(f"加速倍率过大！该图像最多只能加速{max_rate:.2f}倍")
        frames = gif_to_frames(img)
        new_frames = [frames[i] for i in range(0, frame_num, interval)]
        if args.get("back"):
            new_frames.reverse()
        return frames_to_gif(new_frames, duration)


class GrayOperation(ImageOperation):
    def __init__(self):
        super().__init__("gray", ImageType.Any, ImageType.Any, "batch")
        self.help = "将图片转换为灰度图"

    def parse_args(self, args: List[str]):
        assert_and_reply(not args, "该操作不接受参数")
        return None

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        return img.convert("L")


class MidOperation(ImageOperation):
    def __init__(self):
        super().__init__("mid", ImageType.Any, ImageType.Any, "batch")
        self.help = "将图片的一侧对称贴到另一侧，使用方式:\nmid: 左侧贴到右侧\nmid r: 右侧贴到左侧\nmid v: 上侧贴到下侧\nmid v r: 下侧贴到上侧"

    def parse_args(self, args: List[str]) -> dict:
        args = [arg[0].lower() for arg in args]
        assert_and_reply(len(args) <= 2, "最多只支持两个参数")
        mode = "v" if "v" in args else "h"
        if "r" in args:
            mode += "r"
        return {"mode": mode}

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        width, height = img.size
        mode = args["mode"]
        new_img = Image.new("RGBA", (width, height))
        if mode == "h":
            left_img = img.crop((0, 0, width // 2, height))
            new_img.paste(left_img, (0, 0))
            new_img.paste(left_img.transpose(Image.FLIP_LEFT_RIGHT), (width // 2, 0))
        elif mode == "v":
            top_img = img.crop((0, 0, width, height // 2))
            new_img.paste(top_img, (0, 0))
            new_img.paste(top_img.transpose(Image.FLIP_TOP_BOTTOM), (0, height // 2))
        elif mode == "hr":
            right_img = img.crop((width // 2, 0, width, height))
            new_img.paste(right_img.transpose(Image.FLIP_LEFT_RIGHT), (0, 0))
            new_img.paste(right_img, (width // 2, 0))
        else:
            bottom_img = img.crop((0, height // 2, width, height))
            new_img.paste(bottom_img.transpose(Image.FLIP_TOP_BOTTOM), (0, 0))
            new_img.paste(bottom_img, (0, height // 2))
        return new_img


class InvertOperation(ImageOperation):
    def __init__(self):
        super().__init__("invert", ImageType.Any, ImageType.Any, "batch")
        self.help = "将图片颜色反转"

    def parse_args(self, args: List[str]):
        assert_and_reply(not args, "该操作不接受参数")
        return None

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        return ImageOps.invert(img.convert("RGB"))


class RepeatOperation(ImageOperation):
    def __init__(self):
        super().__init__("repeat", ImageType.Any, ImageType.Any, "batch")
        self.help = "将图片重复多次，使用方式:\nrepeat 2 3: 横向重复2次，纵向重复3次"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 2, "需要两个参数")
        ret = {"w": int(args[0]), "h": int(args[1])}
        assert_and_reply(1 <= ret["w"] <= 10 and 1 <= ret["h"] <= 10, "重复次数只能在1-10之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        w_times, h_times = args["w"], args["h"]
        width, height = img.size
        size_limit = 512
        if max(width * w_times, height * h_times) <= size_limit:
            width, height = width * w_times, height * h_times
        elif width * w_times > height * h_times:
            height = size_limit * height * h_times // (width * w_times)
            width = size_limit
        else:
            width = size_limit * width * w_times // (height * h_times)
            height = size_limit
        small_width, small_height = max(1, width // w_times), max(1, height // h_times)
        img = img.resize((small_width, small_height)).convert("RGBA")
        new_img = Image.new("RGBA", (small_width * w_times, small_height * h_times))
        for i in range(w_times):
            for j in range(h_times):
                new_img.paste(img, (i * small_width, j * small_height), img)
        return new_img


class FanOperation(ImageOperation):
    def __init__(self):
        super().__init__("fan", ImageType.Any, ImageType.Animated, "single")
        self.help = "大风车一张图片，使用方式:\nfan: 顺时针旋转\nfan r: 逆时针旋转\nfan 2x: 旋转速度为2倍"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 2, "最多只支持两个参数")
        ret = {"mode": "ccw" if "r" in args else "cw", "speed": 1.0}
        for arg in args:
            if arg.endswith("x"):
                ret["speed"] = float(arg.removesuffix("x"))
        assert_and_reply(0.2 <= ret["speed"] <= 5.0, "旋转速度只能在0.2-5.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        img = img.convert("RGBA")
        if image_type == ImageType.Animated:
            img = img.crop((0, 0, img.width, img.height))
        frame_count = max(1, int(20 / args["speed"]))
        width, height = img.size
        frames = []
        for i in range(frame_count):
            new_img = Image.new("RGBA", (width, height))
            angle = 360 / frame_count * i
            if args["mode"] == "cw":
                angle = -angle
            rotated = img.copy().rotate(angle, expand=False)
            new_img.paste(rotated, (0, 0), rotated)
            frames.append(new_img)
        with TempFilePath("gif") as tmp_path:
            save_transparent_gif(frames, 20, tmp_path)
            return open_image(tmp_path)


class FlowOperation(ImageOperation):
    def __init__(self):
        super().__init__("flow", ImageType.Any, ImageType.Animated, "batch")
        self.help = "添加平移流动效果，使用方式:\nflow: 从左到右\nflow v: 从上到下\nflow r: 从右到左\nflow v r: 从下到上"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 3, "最多只支持三个参数")
        mode = "v" if "v" in args else "h"
        if "r" in args:
            mode += "r"
        ret = {"mode": mode, "speed": 1.0}
        for arg in args:
            if arg.endswith("x"):
                ret["speed"] = float(arg.removesuffix("x"))
        assert_and_reply(0.2 <= ret["speed"] <= 5.0, "流动速度只能在0.2-5.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        img = img.convert("RGBA")
        if image_type == ImageType.Animated:
            img = img.crop((0, 0, img.width, img.height))
        frame_count = max(1, int(20 / args["speed"]))
        width, height = img.size
        frames = []
        mode = args["mode"]
        for i in range(frame_count):
            new_img = Image.new("RGBA", (width, height))
            if mode == "h":
                x = int(i / frame_count * width)
                new_img.paste(img, (x, 0))
                new_img.paste(img, (x - width, 0))
            elif mode == "v":
                y = int(i / frame_count * height)
                new_img.paste(img, (0, y))
                new_img.paste(img, (0, y - height))
            elif mode == "hr":
                x = int(width - i / frame_count * width)
                new_img.paste(img, (x, 0))
                new_img.paste(img, (x - width, 0))
            else:
                y = int(height - i / frame_count * height)
                new_img.paste(img, (0, y))
                new_img.paste(img, (0, y - height))
            frames.append(new_img)
        with TempFilePath("gif") as tmp_path:
            save_transparent_gif(frames, 20, tmp_path)
            return open_image(tmp_path)


class ConcatOperation(ImageOperation):
    def __init__(self):
        super().__init__("concat", ImageType.Multiple, ImageType.Static, "single")
        self.help = "将多张图片拼接成一张，使用方式:\nconcat: 垂直拼接\nconcat h: 水平拼接\nconcat g: 网格拼接"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        mode = "v"
        if "h" in args:
            mode = "h"
        elif "g" in args:
            mode = "g"
        return {"mode": mode}

    def operate(self, imgs, args, image_type=None, frame_idx=0, total_frame=1):
        return concat_images(imgs, args["mode"])


class StackOperation(ImageOperation):
    def __init__(self):
        super().__init__("stack", ImageType.Multiple, ImageType.Animated, "single")
        self.help = "将多张图片堆叠成动图，使用方式:\nstack: 默认fps=20\nstack 10: fps=10"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        ret = {"fps": int(args[0]) if args else 20}
        assert_and_reply(1 <= ret["fps"] <= 50, "fps只能在1-50之间")
        return ret

    def operate(self, imgs, args, image_type=None, frame_idx=0, total_frame=1):
        w, h = imgs[0].size
        frames = [img.resize((w, h)) for img in imgs]
        with TempFilePath("gif") as tmp_path:
            save_transparent_gif(frames, int(1000 / args["fps"]), tmp_path)
            return open_image(tmp_path)


class ExtractOperation(ImageOperation):
    def __init__(self):
        super().__init__("extract", ImageType.Animated, ImageType.Multiple, "single")
        self.help = "将动图拆分成多张图片，使用方式:\nextract: 自动抽帧\nextract 2: 间隔2帧拆分"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        ret = {"interval": int(args[0]) if args else None}
        if ret["interval"] is not None:
            assert_and_reply(1 <= ret["interval"] <= 100, "间隔只能在1-100之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        frames = gif_to_frames(img)
        n_frames = len(frames)
        max_frame_num = 32
        interval = args["interval"]
        if interval:
            if interval >= n_frames:
                raise ReplyException(f"拆分间隔过大！该动图最多只能以{n_frames}帧拆分")
            if n_frames // interval > max_frame_num:
                raise ReplyException(f"拆分间隔过小！该动图最多只能以{n_frames // max_frame_num}帧拆分")
        else:
            interval = max(1, n_frames // max_frame_num)
        return [frames[i] for i in range(0, n_frames, interval)]


class MirageOperation(ImageOperation):
    def __init__(self):
        super().__init__("mirage", ImageType.Multiple, ImageType.Static, "single")
        self.help = "生成幻影坦克图片，使用方式:\nmirage: 倒数第二张为表面图，倒数第一张为隐藏图\nmirage r: 反过来"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        return {"rev": "r" in args}

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        assert_and_reply(len(img) >= 2, "至少需要两张图片")
        surface, hidden = (img[-1], img[-2]) if args["rev"] else (img[-2], img[-1])
        return generate_mirage(surface, hidden)


class BrightenOperation(ImageOperation):
    def __init__(self):
        super().__init__("brighten", ImageType.Any, ImageType.Any, "batch")
        self.help = "调整图片亮度，使用方式:\nbrighten 1.5 / brighten 0.5\n0.0对应黑色图像，1.0对应原图像"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个参数")
        ret = {"ratio": float(args[0])}
        assert_and_reply(0.0 <= ret["ratio"] <= 100.0, "亮度参数只能在0.0-100.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return ImageEnhance.Brightness(img.convert("RGBA")).enhance(args["ratio"])


class ContrastOperation(ImageOperation):
    def __init__(self):
        super().__init__("contrast", ImageType.Any, ImageType.Any, "batch")
        self.help = "调整图片对比度，使用方式:\ncontrast 1.5 / contrast 0.5"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个参数")
        ret = {"ratio": float(args[0])}
        assert_and_reply(0.0 <= ret["ratio"] <= 100.0, "对比度参数只能在0.0-100.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return ImageEnhance.Contrast(img.convert("RGBA")).enhance(args["ratio"])


class SharpenOperation(ImageOperation):
    def __init__(self):
        super().__init__("sharpen", ImageType.Any, ImageType.Any, "batch")
        self.help = "调整图片锐度，使用方式:\nsharpen 1.5 / sharpen 0.5"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个参数")
        ret = {"ratio": float(args[0])}
        assert_and_reply(0.0 <= ret["ratio"] <= 100.0, "锐度参数只能在0.0-100.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return ImageEnhance.Sharpness(img.convert("RGBA")).enhance(args["ratio"])


class SaturateOperation(ImageOperation):
    def __init__(self):
        super().__init__("saturate", ImageType.Any, ImageType.Any, "batch")
        self.help = "调整图片饱和度，使用方式:\nsaturate 1.5 / saturate 0.5"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) == 1, "需要一个参数")
        ret = {"ratio": float(args[0])}
        assert_and_reply(0.01 <= ret["ratio"] <= 100.0, "饱和度参数只能在0.01-100.0之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return ImageEnhance.Color(img.convert("RGBA")).enhance(args["ratio"])


class BlurOperation(ImageOperation):
    def __init__(self):
        super().__init__("blur", ImageType.Any, ImageType.Any, "batch")
        self.help = "对图片进行模糊处理，使用方式:\nblur\nblur 5"

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(len(args) <= 1, "最多只支持一个参数")
        ret = {"radius": int(args[0]) if args else 3}
        assert_and_reply(1 <= ret["radius"] <= 32, "模糊半径只能在1-32之间")
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        return img.convert("RGBA").filter(ImageFilter.GaussianBlur(radius=args["radius"]))


class CropOperation(ImageOperation):
    def __init__(self):
        super().__init__("crop", ImageType.Any, ImageType.Any, "batch")
        self.help = """
裁剪图片，使用方式:
crop 100x100: 裁剪图片100x100中间部分
crop 0.5x0.5: 裁剪图片中心长宽为原来50%的部分
crop 100x100 l: 裁剪图片100x100左边部分(lrtb:左右上下)
crop 100x100 50x50: 裁剪图片100x100，相对左上角偏移(50,50)px
crop l0.1 t0.2 裁剪掉图片左边10%，上边20%部分
""".strip()

    def parse_args(self, args: List[str]) -> dict:
        assert_and_reply(1 <= len(args) <= 4, "参数数量错误")

        def s_to_i_or_f(s):
            if "." in s:
                return float(s)
            if "%" in s:
                return float(s.replace("%", "")) / 100.0
            return int(s)

        ret = {}
        if "x" in args[0]:
            ret["type"] = 1
            ret["size"] = tuple(map(s_to_i_or_f, args[0].split("x")))
            if len(args) == 2:
                if "x" in args[1]:
                    ret["offset"] = tuple(map(s_to_i_or_f, args[1].split("x")))
                else:
                    assert_and_reply(args[1] in ALIGN_MAP, f"指定位置错误，必须是{list(ALIGN_MAP.keys())}中的一个")
                    ret["align"] = args[1].strip()
        else:
            ret["type"] = 2
            ret["border"] = {}
            for arg in args:
                arg = arg.strip()
                assert_and_reply(arg[0] in "lrtb", "裁剪方向错误，必须是(l,r,t,b)中的一个")
                ret["border"][arg[0]] = s_to_i_or_f(arg[1:])
        return ret

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        w, h = img.size

        def getlen(l, ref):
            return int(l * ref) if isinstance(l, float) else l

        def getsize(size):
            return getlen(size[0], w), getlen(size[1], h)

        x1, y1, x2, y2, cw, ch = 0, 0, w, h, w, h
        if args["type"] == 1:
            cw, ch = getsize(args["size"])
            if "offset" in args:
                x1, y1 = getsize(args["offset"])
            else:
                x1, y1 = crop_by_align((w, h), (cw, ch), args.get("align", "c"))[:2]
            x2, y2 = x1 + cw, y1 + ch
        else:
            if "l" in args["border"]:
                x1 = getlen(args["border"]["l"], w)
            if "r" in args["border"]:
                x2 = w - getlen(args["border"]["r"], w)
            if "t" in args["border"]:
                y1 = getlen(args["border"]["t"], h)
            if "b" in args["border"]:
                y2 = h - getlen(args["border"]["b"], h)
            cw, ch = x2 - x1, y2 - y1
        bbox_str = f"[({x1},{y1})->({x2},{y2}) {cw}x{ch}]"
        assert_and_reply(x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h, f"裁剪区域{bbox_str}超出原图像({w}x{h})")
        assert_and_reply(cw > 0, f"裁剪区域{bbox_str}宽度错误")
        assert_and_reply(ch > 0, f"裁剪区域{bbox_str}高度错误")
        return img.crop((x1, y1, x2, y2))


class DemirageOperation(ImageOperation):
    def __init__(self):
        super().__init__("demirage", ImageType.Static, ImageType.Multiple, "single")
        self.help = "提取幻影坦克图片的表图和底图"

    def parse_args(self, args: List[str]):
        assert_and_reply(not args, "该操作不接受参数")
        return None

    def operate(self, img, args, image_type=None, frame_idx=0, total_frame=1):
        surface = Image.new("RGBA", img.size, (255, 255, 255, 255))
        hidden = Image.new("RGBA", img.size, (0, 0, 0, 255))
        surface.paste(img, (0, 0), img)
        hidden.paste(img, (0, 0), img)
        return [surface.convert("RGB"), hidden.convert("RGB")]


class CutoutOperation(ImageOperation):
    def __init__(self):
        super().__init__("cutout", ImageType.Any, ImageType.Any, "single")
        self.help = """
抠图，使用方式:
cutout: 洪水算法抠图（适合纯色背景），容差默认20
cutout 50: 洪水算法抠图，容差50
cutout ai: AI抠图（当前未启用）
""".strip()

    def parse_args(self, args: List[str]) -> dict:
        ret = {"method": "floodfill", "tolerance": 20}
        for arg in args:
            if arg in ["floodfill", "ai"]:
                ret["method"] = arg
            elif arg.isdigit():
                ret["tolerance"] = int(arg)
                assert_and_reply(0 <= ret["tolerance"] <= 255, "容差值只能在0-255之间")
        method_limit = {
            "floodfill": parse_cfg_num(config.get("input_res_limit.cutout.floodfill")),
            "ai": parse_cfg_num(config.get("input_res_limit.cutout.ai")),
        }
        self.input_limit = method_limit.get(ret["method"], self.input_limit)
        return ret

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        frames = gif_to_frames(img) if is_animated(img) else [img]
        if args["method"] == "ai":
            raise ReplyException("AI抠图暂未启用")
        frames = cutout_image(frames, args["tolerance"]).image
        if not isinstance(frames, list):
            frames = [frames]
        if is_animated(img):
            return frames_to_gif(frames, get_gif_duration(img))
        return frames[0]


class ShrinkOperation(ImageOperation):
    def __init__(self):
        super().__init__("shrink", ImageType.Any, ImageType.Any, "single")
        self.help = "裁剪透明，使用方式:\nshrink\nshrink 50\nshrink 10 +10"

    def parse_args(self, args: List[str]) -> dict:
        ret = {"alpha_threshold": 10, "edge": 0}
        assert_and_reply(len(args) <= 2, "最多只支持两个参数")
        for arg in args:
            if "+" in arg:
                ret["edge"] = int(arg.replace("+", ""))
                assert_and_reply(0 <= ret["edge"] <= 100, "扩展像素只能在0-100之间")
            elif arg.isdigit():
                ret["alpha_threshold"] = int(arg)
                assert_and_reply(0 <= ret["alpha_threshold"] <= 255, "透明度阈值只能在0-255之间")
        return ret

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        frames = gif_to_frames(img) if is_animated(img) else [img]
        frames = shrink_image(frames, args["alpha_threshold"], args["edge"]).image
        if not isinstance(frames, list):
            frames = [frames]
        if is_animated(img):
            return frames_to_gif(frames, get_gif_duration(img))
        return frames[0]


class BackgroundOperation(ImageOperation):
    def __init__(self):
        super().__init__("bg", ImageType.Any, ImageType.Any, "batch")
        self.help = "为图片添加背景色，使用方式:\nbg\nbg 255 255 255\nbg #ff00ff"

    def parse_args(self, args: List[str]) -> dict:
        ret = {"color": (255, 255, 255)}
        assert_and_reply(not args or len(args) == 1 or len(args) == 3, "需要一个(颜色代码)或三个(RGB)参数")
        if len(args) == 1:
            ret["color"] = color_code_to_rgb(args[0])[:3]
        elif len(args) == 3:
            r, g, b = int(args[0]), int(args[1]), int(args[2])
            assert_and_reply(0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255, "RGB颜色值必须在0-255之间")
            ret["color"] = (r, g, b)
        return ret

    def operate(self, img, args=None, image_type=None, frame_idx=0, total_frame=1):
        bg = Image.new("RGBA", img.size, args["color"] + (255,))
        bg.alpha_composite(img.convert("RGBA"))
        return bg


def register_all_ops():
    for obj in list(globals().values()):
        if isinstance(obj, type) and issubclass(obj, ImageOperation) and obj is not ImageOperation:
            obj()


register_all_ops()