import argparse
import colorsys
import os
import shlex
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

from src.utils import *

from .ops import ImageOperation, ImageType

config = Config("imgtool")
logger = get_logger("imgtool")
file_db = get_file_db("data/imgtool/db.json")
cd = ColdDown(file_db, logger)
gbl = get_group_black_list(file_db, logger, "imgtool")

IMAGE_LIST_CLEAN_INTERVAL_CFG = config.item("image_list_clean_interval")
MULTI_IMAGE_MAX_NUM_CFG = config.item("multi_image_max_num")


# ======================= 逻辑处理 ======================= #


async def get_reply_fst_image(ctx: HandlerContext, return_url=False):
    """从回复消息里拿第一张图，失败就直接结束不回。"""
    img_url = await ctx.aget_image_urls(return_first=True)
    if return_url:
        return img_url
    try:
        return await download_image(img_url)
    except Exception:
        logger.print_exc(f"获取图片 {img_url} 失败")
        await ctx.asend_reply_msg("获取图片失败")
        raise NoReplyException()


def get_image_list(user_id):
    """取用户当前图片列表，过期会自动清空。"""
    user_id = str(user_id)
    image_list = file_db.get("image_list", {})
    image_list_edit_time = file_db.get("image_list_edit_time", {})
    if user_id not in image_list_edit_time:
        image_list[user_id] = []
        image_list_edit_time[user_id] = datetime.now().timestamp()
        file_db.set("image_list", image_list)
        file_db.set("image_list_edit_time", image_list_edit_time)
        return image_list
    last_edit_time = datetime.fromtimestamp(image_list_edit_time[user_id])
    if (datetime.now() - last_edit_time).total_seconds() > IMAGE_LIST_CLEAN_INTERVAL_CFG.get():
        logger.info("用户 %s 的图片列表已过期", user_id)
        image_list[user_id] = []
        file_db.set("image_list", image_list)
    image_list_edit_time[user_id] = datetime.now().timestamp()
    file_db.set("image_list_edit_time", image_list_edit_time)
    logger.info("获取用户 %s 的图片列表, 共有 %s 张图片", user_id, len(image_list.get(user_id, [])))
    return image_list


async def add_image_to_list(ctx: HandlerContext, reply=True):
    """把消息里的图推进用户图片列表。"""
    args = ctx.get_args()
    user_id = str(ctx.user_id)
    image_list = get_image_list(user_id)
    max_num = MULTI_IMAGE_MAX_NUM_CFG.get()
    img_urls = await ctx.aget_image_urls(min_count=1, max_count=None)
    assert_and_reply(
        len(image_list[user_id]) + len(img_urls) <= max_num,
        f"图片列表已满，当前有{len(image_list[user_id])}张图片，最多只能处理{max_num}张图片",
    )
    if "r" in args:
        img_urls = img_urls[::-1]
    image_list[user_id].extend(img_urls)
    file_db.set("image_list", image_list)
    logger.info("用户 %s 向图片列表添加了 %s 张图片，共有 %s 张", user_id, len(img_urls), len(image_list[user_id]))
    if reply:
        return await ctx.asend_reply_msg(f"成功添加{len(img_urls)}张图片，当前有{len(image_list[user_id])}张图片")


async def pop_image_from_list(ctx: HandlerContext, reply=True):
    """弹出图片列表最后一张。"""
    user_id = str(ctx.user_id)
    image_list = get_image_list(user_id)
    assert_and_reply(image_list[user_id], "图片列表为空")
    img = image_list[user_id].pop()
    file_db.set("image_list", image_list)
    logger.info("用户 %s 从图片列表中弹出图片, 剩余 %s 张", user_id, len(image_list[user_id]))
    if reply:
        return await ctx.asend_reply_msg(f"{await get_image_cq(img)}移除该图片，剩余{len(image_list[user_id])}张图片")


async def clear_image_list(ctx: HandlerContext, reply=True):
    """清空用户图片列表。"""
    user_id = str(ctx.user_id)
    image_list = get_image_list(user_id)
    pre_len = len(image_list[user_id])
    image_list[user_id].clear()
    file_db.set("image_list", image_list)
    logger.info("用户 %s 清空了图片列表, 之前有 %s 张图片", user_id, pre_len)
    if reply:
        return await ctx.asend_reply_msg(f"清空列表中 {pre_len} 张图片")


async def reverse_image_list(ctx: HandlerContext, reply=True):
    """翻转用户图片列表。"""
    user_id = str(ctx.user_id)
    image_list = get_image_list(user_id)
    image_list[user_id].reverse()
    file_db.set("image_list", image_list)
    logger.info("用户 %s 翻转了图片列表", user_id)
    if reply:
        return await ctx.asend_reply_msg(f"翻转成功，当前列表有{len(image_list[user_id])}张图片")


async def get_multi_images(ctx: HandlerContext):
    """从消息或图片列表取出要处理的图。"""
    max_num = MULTI_IMAGE_MAX_NUM_CFG.get()
    img_urls = await ctx.aget_image_urls(min_count=None, max_count=max_num)
    if not img_urls:
        user_id = str(ctx.user_id)
        img_urls = get_image_list(user_id).get(user_id, [])
        assert_and_reply(
            img_urls,
            "请指定要操作的图片！\n方法1. 回复包含单张、多张图片的消息、折叠转发消息\n方法2. 使用图片列表，请使用 /img push 回复包含图片以添加图片",
        )
    imgs = []
    for img_url in img_urls:
        try:
            imgs.append(await download_image(img_url))
        except Exception:
            logger.print_exc(f"获取图片 {img_url} 失败")
            await ctx.asend_reply_msg(f"获取图片 {img_url} 失败")
            raise NoReplyException()
    return imgs[0] if len(imgs) == 1 else imgs


async def operate_image(ctx: HandlerContext):
    """按参数序列对图片做一组操作。"""
    args = ctx.get_args().strip().split()
    all_op_names = ImageOperation.all_ops.keys()
    assert_and_reply(
        args,
        f"操作序列不能为空！\n使用方式: (回复一张图片) /img 操作1 参数1 操作2 参数2 ...\n可用的操作: {', '.join(all_op_names)}\n使用 /img help 操作名 获取某个操作的帮助",
    )
    ops = []
    for arg in args:
        if arg in all_op_names:
            ops.append((ImageOperation.all_ops[arg], []))
        else:
            assert_and_reply(ops, f"未指定初始操作, 可用的操作: {', '.join(all_op_names)}")
            ops[-1][1].append(arg)
    logger.info("请求图片操作 %s 序列: %s", args, [(op.name, op_args) for op, op_args in ops])
    assert_and_reply(ops, f"未指定操作, 可用的操作: {', '.join(all_op_names)}")
    assert_and_reply(len(ops) <= 10, "操作过多, 最多支持10个操作")
    for i in range(1, len(ops)):
        pre_type = ops[i - 1][0].output_type
        cur_type = ops[i][0].input_type
        assert_and_reply(
            pre_type.check_type(cur_type),
            f"第{i}个操作 {ops[i-1][0].name} 的输出类型 {pre_type} 与 第{i+1}个操作 {ops[i][0].name} 的输入类型 {cur_type} 不匹配",
        )
    img = await get_multi_images(ctx)
    img_num = 1 if isinstance(img, Image.Image) else len(img)
    img_type = ImageType.get_type(img)
    first_input_type = ops[0][0].input_type
    if img_num == 1:
        assert_and_reply(first_input_type.check_img(img), f"初始图片类型不匹配, 需要 {first_input_type}, 实际为 {img_type}")
    elif first_input_type != ImageType.Multiple:
        for i, item in enumerate(img):
            assert_and_reply(
                first_input_type.check_img(item),
                f"第{i+1}张图片类型不匹配, 需要 {first_input_type}, 实际为 {ImageType.get_type(item)}",
            )
    for i, (op, op_args) in enumerate(ops):
        try:
            img = await run_in_pool(op, img, op_args)
        except Exception as e:
            logger.print_exc(f"执行第{i+1}个图片操作 {op.name} 失败")
            raise ReplyException(f"执行第{i+1}个图片操作 {op.name} 失败: {e}") from e
    await clear_image_list(ctx, reply=False)
    logger.info("%s个图片操作全部执行完毕", len(ops))
    if isinstance(img, list):
        msgs = [f"{await get_image_cq(item)}#{i}" for i, item in enumerate(img)]
        return await ctx.asend_fold_msg(msgs)
    return await ctx.asend_reply_msg(await get_image_cq(img))


# ======================= 指令处理 ======================= #

img_op = CmdHandler(["/img"], logger, priority=1)
img_op.check_cdrate(cd).check_wblist(gbl)


@img_op.handle()
async def _(ctx: HandlerContext):
    await ctx.block(f"{ctx.user_id}", 5)
    await operate_image(ctx)


img_push = CmdHandler(["/img push", "/imgpush"], logger, priority=1)
img_push.check_cdrate(cd).check_wblist(gbl)


@img_push.handle()
async def _(ctx: HandlerContext):
    await add_image_to_list(ctx)


img_pop = CmdHandler(["/img pop", "/imgpop"], logger, priority=1)
img_pop.check_cdrate(cd).check_wblist(gbl)


@img_pop.handle()
async def _(ctx: HandlerContext):
    await pop_image_from_list(ctx)


img_clear = CmdHandler(["/img clear", "/imgclear"], logger, priority=1)
img_clear.check_cdrate(cd).check_wblist(gbl)


@img_clear.handle()
async def _(ctx: HandlerContext):
    await clear_image_list(ctx)


img_reverse = CmdHandler(["/img rev", "/imgrev"], logger, priority=1)
img_reverse.check_cdrate(cd).check_wblist(gbl)


@img_reverse.handle()
async def _(ctx: HandlerContext):
    await reverse_image_list(ctx)


img_help = CmdHandler(["/img help", "/imghelp", "/imgh"], logger, priority=1, disable_help=True)
img_help.check_cdrate(cd).check_wblist(gbl)


@img_help.handle()
async def _(ctx: HandlerContext):
    ops = ImageOperation.all_ops
    op_name = ctx.get_args().strip()
    assert_and_reply(op_name, f"请输入要查找帮助的操作名，可用的操作: {', '.join(ops.keys())}")
    op = ops.get(op_name)
    assert_and_reply(op, f"未找到操作 {op_name}, 可用的操作: {', '.join(ops.keys())}")
    msg = f"【{op.name}】\n{op.input_type} -> {op.output_type}\n{op.help}"
    return await ctx.asend_reply_msg(msg.strip())


img_check = CmdHandler(["/img check", "/img_check", "/img info", "/img_info"], logger, priority=1)
img_check.check_cdrate(cd).check_wblist(gbl)


@img_check.handle()
async def _(ctx: HandlerContext):
    data_list = await ctx.aget_image_datas()
    msg = ""
    for i, data in enumerate(data_list):
        with TempFilePath("png") as path:
            msg += f"\n\n【图片{i+1}】"
            url = data.get("url")
            try:
                await download_file(url, path)
                img = Image.open(path)
                msg += f"\n分辨率: {img.width}x{img.height}"
                if is_animated(img):
                    msg += f"\n长度: {img.n_frames}帧"
                    if not img.info.get("duration", 0):
                        msg += "\n帧间隔/FPS: 未知"
                    else:
                        msg += f"\n帧间隔: {img.info['duration']}ms"
                        msg += f"\nFPS: {1000 / img.info['duration']:.2f}"
                if "file_size" in data:
                    filesize = int(data["file_size"] or 0) or os.path.getsize(path)
                    msg += f"\n文件大小: {get_readable_file_size(filesize)}"
                if url:
                    msg += f"\n链接: {url}"
                if data.get("file_unique"):
                    msg += f"\n图片标识: {data['file_unique']}"
            except Exception as e:
                logger.print_exc(f"获取 {url} 图片信息失败")
                msg += f"\n无法获取图片信息: {get_exc_desc(e)}"
    return await ctx.asend_fold_msg_adaptive(msg.strip())


scan = CmdHandler(["/scan", "/扫描", "/识别"], logger)
scan.check_cdrate(cd).check_wblist(gbl)


@scan.handle()
async def _(ctx: HandlerContext):
    img = await get_reply_fst_image(ctx)
    try:
        from pyzbar.pyzbar import decode
    except ImportError as e:
        raise ReplyException("未安装 pyzbar，无法识别二维码") from e
    res = decode(img)
    assert_and_reply(res, "未发现二维码")
    msg = "\n".join([r.data.decode("utf-8") for r in res])
    return await ctx.asend_reply_msg(f"共识别{len(res)}个条形码/二维码:\n{msg}")


gen_qrcode = CmdHandler(["/qrcode", "/二维码"], logger)
gen_qrcode.check_cdrate(cd).check_wblist(gbl)


@gen_qrcode.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    assert_and_reply(args, "请输入内容")
    try:
        import qrcode
    except ImportError as e:
        raise ReplyException("未安装 qrcode，无法生成二维码") from e
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(args)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    with TempFilePath("png") as tmp_path:
        img.save(tmp_path)
        return await ctx.asend_reply_msg(await get_image_cq(Image.open(tmp_path)))


gen_saying = CmdHandler(["/saying", "/quote", "/语录"], logger)
gen_saying.check_cdrate(cd).check_wblist(gbl).check_group()


@gen_saying.handle()
async def _(ctx: HandlerContext):
    reply_msg = await ctx.aget_reply_msg()
    assert_and_reply(reply_msg, "无法获取回复消息")
    reply_cqs = extract_cq_code(reply_msg)
    sender = ctx.get_reply_sender()
    if "forward" in reply_cqs:
        content = reply_cqs["forward"][0].get("content") or []
        if content:
            reply_msg_obj = content[0]
            reply_msg = reply_msg_obj.get("message") or reply_msg
    reply_user_id = 0
    reply_user_name = ""
    if isinstance(sender, dict):
        reply_user_id = int(sender.get("user_id") or 0)
        reply_user_name = str(sender.get("card") or sender.get("nickname") or "")
    elif sender is not None:
        reply_user_id = int(getattr(sender, "user_id", 0) or 0)
        reply_user_name = str(getattr(sender, "card", None) or getattr(sender, "nickname", None) or "")
    if not reply_user_name:
        reply_user_name = get_user_name_by_event(ctx.event.get("reply") if ctx.event else None)
    text = extract_text(reply_msg).strip()
    assert_and_reply(text, "回复的消息没有文本!")
    text = "「 " + text + " 」"
    line_len = 20
    name_text = "——" + reply_user_name
    avatar = None
    if reply_user_id:
        try:
            avatar = await download_image(await get_avatar_url_large(ctx.bot, reply_user_id))
        except Exception:
            logger.print_exc("下载语录头像失败")
    with Canvas(bg=FillBg(BLACK)) as canvas:
        with HSplit().set_item_align("c").set_content_align("c").set_padding(16).set_sep(16):
            with VSplit().set_item_align("c").set_content_align("c"):
                Spacer(10, 32)
                if avatar is not None:
                    ImageBox(avatar, size=(256, 256)).set_margin(16)
                Spacer(10, 32)
            with VSplit().set_item_align("c").set_content_align("c").set_sep(8):
                font_sz = 48
                TextBox(text, TextStyle(DEFAULT_FONT, font_sz, WHITE), line_count=get_str_line_count(text, line_len) + 1).set_w(font_sz * line_len // 2).set_content_align("l")
                TextBox(name_text, TextStyle(DEFAULT_FONT, font_sz, WHITE)).set_w(font_sz * line_len // 2).set_content_align("r")
            Spacer(16, 16)
    return await ctx.asend_reply_msg(await get_image_cq(await canvas.get_img()))


def color_card(color, additional_text=None):
    if sum(color) > 255 * 3 / 2:
        back_color, front_color = BLACK, WHITE
    else:
        back_color, front_color = WHITE, BLACK
    r, g, b = color
    h, s, l = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h, s, l = int(h * 360), int(s * 100), int(l * 100)
    text_style = TextStyle(DEFAULT_FONT, 20, front_color)
    with VSplit().set_bg(FillBg(back_color)).set_item_align("c").set_content_align("c").set_padding(8).set_sep(4) as card:
        Spacer(128, 128).set_bg(RoundRectBg((*color, 255), 8))
        if additional_text:
            TextBox(additional_text, text_style)
        TextBox(f"#{r:02x}{g:02x}{b:02x}", text_style)
        TextBox(f"rgb({r},{g},{b})", text_style)
        TextBox(f"hsl({h},{s},{l})", text_style)
    return card


color_show = CmdHandler(["/color", "/颜色"], logger)
color_show.check_cdrate(cd).check_wblist(gbl)


@color_show.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    r = g = b = 0
    try:
        if "#" in args:
            args = args.replace("#", "").strip()
            if len(args) == 3:
                args = "".join([c * 2 for c in args])
            r, g, b = int(args[:2], 16), int(args[2:4], 16), int(args[4:], 16)
        elif "hsl" in args:
            h, s, l = args.replace("hsl", "").split()
            r, g, b = colorsys.hls_to_rgb(float(h) / 360, float(l) / 100, float(s) / 100)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        elif "rgbf" in args:
            r, g, b = args.replace("rgbf", "").split()
            r, g, b = int(float(r) * 255), int(float(g) * 255), int(float(b) * 255)
        else:
            r, g, b = args.replace("rgb", "").split()
            r, g, b = int(r), int(g), int(b)
    except Exception:
        logger.print_exc("参数解析失败")
        return await ctx.asend_reply_msg("参数错误，使用示例:\n/color #aabbcc\n/color #abc\n/color hsl 120 50 50\n/color rgb 255 255 255\n/color rgbf 1.0 1.0 1.0")
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    with Canvas(bg=FillBg(WHITE)) as canvas:
        color_card([r, g, b])
    return await ctx.asend_reply_msg(await get_image_cq(await canvas.get_img()))


def _pick_colors(arr: np.ndarray, k: int):
    pixels = arr.reshape((-1, 3))
    quantized = (pixels // 16) * 16 + 8
    counts = Counter(map(tuple, quantized.tolist()))
    return [color for color, _ in counts.most_common(k)]


color_picker = CmdHandler(["/pick", "/取色"], logger, priority=1)
color_picker.check_cdrate(cd).check_wblist(gbl)


@color_picker.handle()
async def _(ctx: HandlerContext):
    img = (await get_reply_fst_image(ctx)).convert("RGB")
    arr = np.array(img)
    top_k = config.get("color_pick_topk") or 10
    args = ctx.get_args().strip()
    if args:
        top_k = int(args)
        assert_and_reply(1 <= top_k <= 20, "取色数量只能在1-20之间")
    colors = await run_in_pool(_pick_colors, arr, top_k)
    with Canvas(bg=FillBg((200, 200, 200, 255))) as canvas:
        with Grid(col_count=5).set_item_align("c").set_content_align("c").set_sep(8):
            for color in colors:
                color_card(color).set_w(180)
    return await ctx.asend_reply_msg(await get_image_cq(await canvas.get_img()))


def convert_video_to_gif(video_path: str, save_path: str, max_fps=10, max_size=256, max_frame_num=200):
    try:
        import ffmpeg
    except ImportError as e:
        raise ReplyException("未安装 ffmpeg-python，无法转换视频") from e

    logger.info("转换视频为GIF: %s", video_path)
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    assert_and_reply(video_stream, "视频没有视频流")
    frame_num = int(float(video_stream.get("nb_frames") or 0) or 0)
    fps_text = video_stream.get("avg_frame_rate") or "10/1"
    num, den = fps_text.split("/")
    fps = float(num) / max(1.0, float(den))
    duration = frame_num / fps if frame_num and fps else float(video_stream.get("duration") or 1)
    max_fps = max(min(max_fps, int(max_frame_num / max(duration, 0.1))), 1)
    width, height = int(video_stream["width"]), int(video_stream["height"])
    if width > height:
        if width > max_size:
            height = int(height * max_size / width)
            width = max_size
    elif height > max_size:
        width = int(width * max_size / height)
        height = max_size
    palette_stream = ffmpeg.input(video_path).filter_multi_output("split")[0].filter("palettegen")
    filtered = ffmpeg.input(video_path).filter("fps", fps=max_fps).filter("scale", width=width, height=-1, flags="lanczos")
    stream = ffmpeg.filter([filtered, palette_stream], "paletteuse")
    ffmpeg.run(ffmpeg.output(stream, save_path), overwrite_output=True, quiet=True)


video_to_gif = CmdHandler(["/gif"], logger)
video_to_gif.check_cdrate(cd).check_wblist(gbl)


@video_to_gif.handle()
async def _(ctx: HandlerContext):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_size", "-s", type=int, default=config.get("video_to_gif.default_max_size"))
    parser.add_argument("--max_fps", "-f", type=int, default=config.get("video_to_gif.default_max_fps"))
    parser.add_argument("--max_frame_num", "-n", type=int, default=config.get("video_to_gif.default_max_frame_num"))
    try:
        args = parser.parse_args(shlex.split(ctx.get_args()))
    except SystemExit as e:
        raise ReplyException(
            "使用方式: (回复一个视频) /gif [--max_size/-s <最大尺寸>] [--max_fps/-f <最大帧率>] [--max_frame_num/-n <最大帧数>]"
        ) from e
    reply_msg = await ctx.aget_reply_msg()
    assert_and_reply(reply_msg, "请回复一条带有视频的消息")
    cqs = extract_cq_code(reply_msg)
    assert_and_reply("video" in cqs, "回复的消息中没有视频")
    video = cqs["video"][0]
    video_url = video.get("url")
    assert_and_reply(video_url, "无法获取视频链接")
    filesize = int(video.get("file_size") or 0)
    size_limit = int(config.get("video_to_gif.size_limit") * 1024 * 1024)
    if filesize:
        assert_and_reply(filesize <= size_limit, "视频文件过大，无法处理")
    with TempFilePath("video") as video_path:
        await download_file(video_url, video_path)
        with TempFilePath("gif") as gif_path:
            await run_in_pool(convert_video_to_gif, video_path, gif_path, args.max_fps, args.max_size, args.max_frame_num)
            return await ctx.asend_reply_msg(await get_image_cq(gif_path))
