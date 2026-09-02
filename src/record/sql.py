from datetime import datetime

import aiosqlite

from src.utils import *

config = Config("record")
logger = get_logger("record")

DB_PATH = "data/record/record.sqlite"
MSG_TABLE_NAME = "msg_{}"

_conn: aiosqlite.Connection | None = None
_created_table_group_ids: set[int] = set()


async def get_conn(group_id: int | list[int]):
    """拿到 sqlite 连接，缺表就建。"""
    global _conn
    if _conn is None:
        create_parent_folder(DB_PATH)
        _conn = await aiosqlite.connect(DB_PATH)
        logger.info("连接sqlite数据库 %s 成功", DB_PATH)

    table_created = False
    if isinstance(group_id, int):
        group_id = [group_id]
    for gid in group_id:
        if gid not in _created_table_group_ids:
            await _conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {MSG_TABLE_NAME.format(gid)} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER,
                    msg_id INTEGER,
                    user_id INTEGER,
                    nickname TEXT,
                    content TEXT
                )
                """
            )
            _created_table_group_ids.add(gid)
            table_created = True
    if table_created:
        await _conn.commit()
    return _conn


async def insert_msg(group_id, time: datetime, msg_id: int, user_id: int, nickname: str, msg: dict):
    """插入一条群消息。"""
    time_ts = time.timestamp()
    content = dumps_json(msg, indent=False)
    conn = await get_conn(group_id)
    insert_query = f"""
        INSERT INTO {MSG_TABLE_NAME.format(group_id)} (time, msg_id, user_id, nickname, content)
        VALUES (?, ?, ?, ?, ?)
    """
    await conn.execute(insert_query, (time_ts, msg_id, user_id, nickname, content))
    await conn.commit()
    logger.debug("插入消息 %s 到 %s 表", msg_id, MSG_TABLE_NAME.format(group_id))


async def insert_msgs(msgs: list):
    """按群批量插入消息。"""
    group_id_msgs: dict[int, list] = {}
    for msg in msgs:
        group_id_msgs.setdefault(msg["group_id"], []).append(msg)

    conn = await get_conn(list(group_id_msgs.keys()))
    for group_id, group_msgs in group_id_msgs.items():
        insert_query = f"""
            INSERT INTO {MSG_TABLE_NAME.format(group_id)} (time, msg_id, user_id, nickname, content)
            VALUES (?, ?, ?, ?, ?)
        """
        values = []
        for msg in group_msgs:
            values.append(
                (
                    msg["time"].timestamp(),
                    msg["msg_id"],
                    msg["user_id"],
                    msg["nickname"],
                    dumps_json(msg["msg"], indent=False),
                )
            )
        await conn.executemany(insert_query, values)
    await conn.commit()


def msg_row_to_ret(row):
    """把 sqlite 行转成消息 dict。"""
    return {
        "id": row[0],
        "time": datetime.fromtimestamp(row[1]),
        "msg_id": row[2],
        "user_id": row[3],
        "nickname": row[4],
        "msg": loads_json(row[5]),
    }


async def query_all_msg(group_id: int):
    """查出一个群的全部消息。"""
    conn = await get_conn(group_id)
    cursor = await conn.execute(f"SELECT * FROM {MSG_TABLE_NAME.format(group_id)}")
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug("获取 %s 表中的所有消息 %s 条", MSG_TABLE_NAME.format(group_id), len(rows))
    return [msg_row_to_ret(row) for row in rows]


async def query_msg_by_range(group_id: int, start_time: datetime, end_time: datetime):
    """按时间范围查消息。"""
    if start_time is None:
        start_time = datetime.fromtimestamp(0)
    if end_time is None:
        end_time = datetime.fromtimestamp(9999999999)
    conn = await get_conn(group_id)
    cursor = await conn.execute(
        f"""
        SELECT * FROM {MSG_TABLE_NAME.format(group_id)}
        WHERE time >= ? AND time <= ?
        """,
        (start_time.timestamp(), end_time.timestamp()),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug(
        "获取 %s 表中的 从 %s 到 %s 的消息 %s 条",
        MSG_TABLE_NAME.format(group_id),
        start_time,
        end_time,
        len(rows),
    )
    return [msg_row_to_ret(row) for row in rows]


async def query_recent_msg(group_id: int, limit: int):
    """查最近若干条消息。"""
    conn = await get_conn(group_id)
    cursor = await conn.execute(
        f"""
        SELECT * FROM {MSG_TABLE_NAME.format(group_id)}
        ORDER BY time DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug("获取 %s 表中的 最近 %s 条消息 %s 条", MSG_TABLE_NAME.format(group_id), limit, len(rows))
    return [msg_row_to_ret(row) for row in rows]


async def query_msg_count(group_id: int, start_time: datetime, end_time: datetime, user_id: int | None = None):
    """统计时间范围内的消息条数，可按用户过滤。"""
    if start_time is None:
        start_time = datetime.fromtimestamp(0)
    if end_time is None:
        end_time = datetime.fromtimestamp(9999999999)
    conn = await get_conn(group_id)
    if user_id is None:
        cursor = await conn.execute(
            f"""
            SELECT COUNT(*) FROM {MSG_TABLE_NAME.format(group_id)}
            WHERE time >= ? AND time <= ?
            """,
            (start_time.timestamp(), end_time.timestamp()),
        )
    else:
        cursor = await conn.execute(
            f"""
            SELECT COUNT(*) FROM {MSG_TABLE_NAME.format(group_id)}
            WHERE time >= ? AND time <= ? AND user_id = ?
            """,
            (start_time.timestamp(), end_time.timestamp(), user_id),
        )
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug("获取 %s 表中的 从 %s 到 %s 的消息数", MSG_TABLE_NAME.format(group_id), start_time, end_time)
    return rows[0][0]


async def query_msg_by_user_id(group_id: int, user_id: int):
    """查某个用户在群里发过的消息。"""
    conn = await get_conn(group_id)
    cursor = await conn.execute(
        f"""
        SELECT * FROM {MSG_TABLE_NAME.format(group_id)}
        WHERE user_id = ?
        """,
        (user_id,),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug("获取 %s 表中的 用户 %s 的消息 %s 条", MSG_TABLE_NAME.format(group_id), user_id, len(rows))
    return [msg_row_to_ret(row) for row in rows]


async def query_msg_before(group_id: int, time: datetime, limit: int):
    """查某个时间点之前的若干条消息。"""
    conn = await get_conn(group_id)
    cursor = await conn.execute(
        f"""
        SELECT * FROM {MSG_TABLE_NAME.format(group_id)}
        WHERE time <= ?
        ORDER BY time DESC
        LIMIT ?
        """,
        (time.timestamp(), limit),
    )
    rows = await cursor.fetchall()
    await cursor.close()
    logger.debug(
        "获取 %s 表中的 时间在 %s 之前的 %s 条消息 %s 条",
        MSG_TABLE_NAME.format(group_id),
        time,
        limit,
        len(rows),
    )
    return [msg_row_to_ret(row) for row in rows]
