import glob
import os
from pathlib import Path

from src.utils import *

config = Config("helper")
logger = get_logger("helper")
file_db = get_file_db("data/helper/db.json")
gbl = get_group_black_list(file_db, logger, "helper")
cd = ColdDown(file_db, logger)

HELP_DOCS_PATH = "helps/{name}.md"
HELP_IMG_SCALE = config.item("img_scale")
HELP_IMG_WIDTH = config.item("img_width")
HELP_IMG_INTERSECT = config.item("img_intersect")


# ======================= 逻辑处理 ======================= #


def _list_help_docs() -> list[tuple[str, str]]:
    """扫 helps 目录，返回 (模块名, 标题) 列表。"""
    items: list[tuple[str, str]] = []
    for path in glob.glob(HELP_DOCS_PATH.format(name="*")):
        try:
            if path.endswith("main.md"):
                continue
            first_line = Path(path).read_text(encoding="utf-8").splitlines()[0].strip()
            desc = first_line.lstrip("#").strip()
            if desc.endswith(")") and "(" in desc:
                desc = desc[: desc.rfind("(")].strip()
            items.append((Path(path).stem, desc))
        except Exception:
            continue
    items.sort(key=lambda x: x[0])
    return items


# ======================= 指令处理 ======================= #

help_cmd = CmdHandler(["/help", "/帮助"], logger, block=True, disable_help=True)
help_cmd.check_wblist(gbl).check_cdrate(cd)


@help_cmd.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()
    docs = _list_help_docs()
    names = [name for name, _ in docs]

    if not args or args not in names:
        service_list = "\n".join(f"{name} - {desc}" for name, desc in docs)
        template = str(config.get("template"))
        if "{service_list}" in template:
            template = template.format(service_list=service_list.strip())
        return await ctx.asend_fold_msg_adaptive(template.strip(), need_reply=False)

    try:
        from src.draw.markdown import markdown_to_image

        doc_path = Path(HELP_DOCS_PATH.format(name=args))
        doc_mtime = os.path.getmtime(doc_path)
        cache_mtime = file_db.get("help_img_cache_mtime", {})
        cache_path = create_parent_folder(f"data/helper/cache/{args}.png")
        if Path(cache_path).exists() and doc_mtime <= cache_mtime.get(args, 0):
            return await ctx.asend_reply_msg(await get_image_cq(cache_path, low_quality=True))

        logger.info("缓存 %s 帮助文档不存在或已过期，重新渲染", args)
        image = await markdown_to_image(
            doc_path.read_text(encoding="utf-8"),
            width=int(HELP_IMG_WIDTH.get()),
            scale=float(HELP_IMG_SCALE.get()),
            intersect=int(HELP_IMG_INTERSECT.get()),
        )
        image.save(cache_path)
        cache_mtime[args] = doc_mtime
        file_db.set("help_img_cache_mtime", cache_mtime)
        return await ctx.asend_reply_msg(await get_image_cq(image, low_quality=True))
    except Exception:
        logger.print_exc(f"渲染 {args} 帮助文档失败")
        return await ctx.asend_reply_msg("帮助文档渲染失败")
