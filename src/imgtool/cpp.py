import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

from src.utils import *

logger = get_logger("imgtool")
CPP_PATH = "data/imgtool/imgtool-cpp"


@dataclass
class CppImageOutput:
    image: Image.Image | List[Image.Image]
    extra_info: dict = field(default_factory=dict)


def execute_imgtool_cpp(image: Image.Image | List[Image.Image], command: str, *args) -> CppImageOutput:
    """调用 imgtool-cpp 处理图片。"""
    is_single_frame = isinstance(image, Image.Image)
    if is_single_frame:
        image = [image]
    ret = []
    w, h = image[0].size
    n = len(image)
    with TempFilePath("input") as input_path:
        with TempFilePath("output") as output_path:
            with open(input_path, "wb") as f:
                f.write(int(n).to_bytes(4, sys.byteorder))
                f.write(int(h).to_bytes(4, sys.byteorder))
                f.write(int(w).to_bytes(4, sys.byteorder))
                for i in range(n):
                    frame = image[i].convert("RGBA")
                    f.write(frame.tobytes("raw", "RGBA"))

            cli_path = CPP_PATH
            logger.info(f"调用imgtool-cpp程序: {command} " + " ".join(map(str, args)) + f" 输入尺寸: {n}x{w}x{h}")
            assert_and_reply(os.path.exists(cli_path), "imgtool-cpp程序不存在，请使用scripts/compile_imgtool_cpp.sh编译")
            cmd = f"{cli_path} {input_path} {output_path} {command} " + " ".join(map(str, args))
            assert_and_reply(os.system(cmd) == 0, "调用imgtool-cpp程序失败")

            with open(output_path, "rb") as f:
                n = int.from_bytes(f.read(4), sys.byteorder)
                h = int.from_bytes(f.read(4), sys.byteorder)
                w = int.from_bytes(f.read(4), sys.byteorder)
                for i in range(n):
                    frame = Image.new("RGBA", (w, h))
                    frame.frombytes(f.read(w * h * 4), "raw", "RGBA")
                    ret.append(frame)
                extra_info = {}
                try:
                    extra_n = int.from_bytes(f.read(4), sys.byteorder)
                    extra_info = loads_json(f.read(extra_n)) if extra_n > 0 else {}
                except Exception as e:
                    logger.warning(f"imgtool-cpp程序返回的extra_info数据解析失败: {get_exc_desc(e)}")
                    extra_info = {}
            logger.info(f"imgtool-cpp程序执行完毕，输出尺寸: {n}x{w}x{h}，额外返回: {extra_info}")

    return CppImageOutput(image=ret[0] if is_single_frame else ret, extra_info=extra_info)


def cutout_image(image: Image.Image | List[Image.Image], tolerance: int) -> CppImageOutput:
    """抠图。"""
    return execute_imgtool_cpp(image, "cutout", tolerance)


def shrink_image(image: Image.Image | List[Image.Image], alpha_threshold: int, edge: int) -> CppImageOutput:
    """按透明边裁掉空白，失败时用 Python 备用实现。"""
    try:
        return execute_imgtool_cpp(image, "shrink", alpha_threshold, edge)
    except Exception as e:
        logger.warning(f"imgtool-cpp程序shrink命令执行失败，使用备用实现: {get_exc_desc(e)}")
        is_single = isinstance(image, Image.Image)
        frames = [image] if is_single else list(image)
        outs = []
        extra_ret = {}
        for frame in frames:
            image_array = np.array(frame.convert("RGBA"))
            alpha = image_array[:, :, 3]
            non_blank_rows = np.where(np.any(alpha >= alpha_threshold, axis=1))[0]
            non_blank_columns = np.where(np.any(alpha >= alpha_threshold, axis=0))[0]
            left = top = 0
            if non_blank_rows.size > 0 and non_blank_columns.size > 0:
                left = int(non_blank_columns[0])
                right = int(non_blank_columns[-1])
                top = int(non_blank_rows[0])
                bottom = int(non_blank_rows[-1])
                cropped = Image.fromarray(image_array[top : bottom + 1, left : right + 1])
            else:
                cropped = Image.fromarray(image_array)
            w, h = cropped.size
            new_image = Image.new("RGBA", (w + 2 * edge, h + 2 * edge), (0, 0, 0, 0))
            new_image.paste(cropped, (edge, edge))
            outs.append(new_image)
            extra_ret = {"bbox": (left, top, w, h)}
        return CppImageOutput(image=outs[0] if is_single else outs, extra_info=extra_ret)
