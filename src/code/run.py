import asyncio
import re

import aiohttp

from src.utils import *

config = Config("code")

codeType = {
    "py": ["python", "py"],
    "cpp": ["cpp", "cpp"],
    "java": ["java", "java"],
    "php": ["php", "php"],
    "js": ["javascript", "js"],
    "c": ["c", "c"],
    "c#": ["csharp", "cs"],
    "go": ["go", "go"],
    "asm": ["assembly", "asm"],
    "ats": ["ats", "dats"],
    "bash": ["bash", "sh"],
    "clisp": ["clisp", "lsp"],
    "clojure": ["clojure", "clj"],
    "cobol": ["cobol", "cob"],
    "coffeescript": ["coffeescript", "coffee"],
    "crystal": ["crystal", "cr"],
    "D": ["D", "d"],
    "elixir": ["elixir", "ex"],
    "elm": ["elm", "elm"],
    "erlang": ["erlang", "erl"],
    "fsharp": ["fsharp", "fs"],
    "groovy": ["groovy", "groovy"],
    "guile": ["guile", "scm"],
    "hare": ["hare", "ha"],
    "haskell": ["haskell", "hs"],
    "idris": ["idris", "idr"],
    "julia": ["julia", "jl"],
    "kotlin": ["kotlin", "kt"],
    "lua": ["lua", "lua"],
    "mercury": ["mercury", "m"],
    "nim": ["nim", "nim"],
    "nix": ["nix", "nix"],
    "ocaml": ["ocaml", "ml"],
    "pascal": ["pascal", "pp"],
    "perl": ["perl", "pl"],
    "raku": ["raku", "raku"],
    "ruby": ["ruby", "rb"],
    "rust": ["rust", "rs"],
    "sac": ["sac", "sac"],
    "scala": ["scala", "scala"],
    "swift": ["swift", "swift"],
    "typescript": ["typescript", "ts"],
    "zig": ["zig", "zig"],
    "plaintext": ["plaintext", "txt"],
}


async def run(strcode):
    """把代码丢给 glot.io 跑，返回 stdout/stderr。"""
    strcode = strcode.replace("&amp;", "&").replace("&#91;", "[").replace("&#93;", "]")
    try:
        pattern = r"(" + "|".join(codeType.keys()) + r")\b ?(.*)\n((?:.|\n)+)"
        a = re.match(pattern, strcode)
        lang, stdin, code = a.group(1), a.group(2).replace(" ", "\n"), a.group(3)
    except Exception:
        return f"目前仅支持{'/'.join(codeType.keys())}"

    data_json = {
        "files": [
            {
                "name": f"main.{codeType[lang][1]}",
                "content": code,
            }
        ],
        "stdin": stdin,
        "command": "",
    }
    headers = {
        "Authorization": f"Token {config.get('token')}",
        "content-type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=config.get("timeout"))
    try:
        async with get_client_session().post(
            url=f"https://glot.io/run/{codeType[lang][0]}?version=latest",
            headers=headers,
            json=data_json,
            timeout=timeout,
            ssl=False,
        ) as res:
            status = res.status
            if status == 200:
                data = await res.json()
                stderr = data.get("stderr") or ""
                return data.get("stdout", "") + ("\n---\n" + stderr if stderr else "")
            text = await res.text()
            raise Exception(f"请求失败({status}):{text}")
    except (asyncio.TimeoutError, TimeoutError):
        raise Exception("请求超时")
    except Exception as e:
        if str(e).startswith("请求失败") or str(e) == "请求超时":
            raise
        raise Exception(f"请求失败: {type(e).__name__} {e}")
