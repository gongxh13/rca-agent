from typing import Any

import pandas as pd


def to_iso_shanghai(ts: Any) -> str:
    """
    将时间戳统一格式化为带 Asia/Shanghai 时区的 ISO 8601 字符串。

    - 输入可以是 pd.Timestamp、numpy datetime64 或其他可被 pd.Timestamp 解析的类型。
    - 如果没有时区信息，则假定为 Asia/Shanghai 并本地化。
    - 如果有其他时区信息，则统一转换为 Asia/Shanghai。
    """
    if isinstance(ts, pd.Timestamp):
        timestamp = ts
    else:
        timestamp = pd.Timestamp(ts)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    else:
        timestamp = timestamp.tz_convert("Asia/Shanghai")

    return timestamp.isoformat()


