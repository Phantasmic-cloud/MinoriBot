from src.utils import *

from .run import run

config = Config("code")
logger = get_logger("code")
file_db = get_file_db("data/code/db.json")
cd = ColdDown(file_db, logger)
gbl = get_group_black_list(file_db, logger, "code")


# ======================= 指令处理 ======================= #

runcode = CmdHandler(["/code", "/run"], logger)
runcode.check_cdrate(cd).check_wblist(gbl)


@runcode.handle()
async def _(ctx: HandlerContext):
    code = ctx.get_args().strip()
    assert_and_reply(code, "请输入要运行的代码")
    logger.info("运行代码: %s", code)
    res = await run(code)
    logger.info("运行结果: %s", res)
    return await ctx.asend_fold_msg_adaptive(res)
