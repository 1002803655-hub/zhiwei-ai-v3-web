# -*- coding: utf-8 -*-
"""Private ephemeral data attachments for the Lovis composer.

Data remains in the Streamlit session, never in browser chat history.
MAT files are read locally; operational CSV rows may be sent to Dify only
after a *separate, explicit* numeric-data opt-in.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from datetime import datetime
from typing import Any

MAT_MAX_BYTES = 25 * 1024 * 1024
CSV_MAX_BYTES = 64 * 1024
CSV_MAX_ROWS = 200
CSV_FIELDS = ("time", "current_a", "current_b", "current_c", "bearing_temp")


def source_rate_hint(name: str) -> int:
    # Official CWRU normal baseline files 97–100 use 48 kHz.
    # Filename does not verify origin: user must confirm.
    return 48000 if re.fullmatch(r"(97|98|99|100)\.mat", name, re.I) else 12000


def validate_trend_csv(data: bytes) -> tuple[str, int]:
    """Validate/sanitize the CSV before any optional API transmission."""
    if len(data) > CSV_MAX_BYTES:
        raise ValueError("趋势CSV超过64KB；请分段上传最多200条记录")
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV请另存为UTF-8编码") from exc
    try:
        rows = list(csv.reader(io.StringIO(decoded, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("CSV列或引号格式错误") from exc
    if not rows:
        raise ValueError("CSV为空")
    headers = tuple(c.strip() for c in rows[0])
    if headers != CSV_FIELDS:
        raise ValueError("CSV需依次包含time,current_a,current_b,current_c,bearing_temp五列")
    if not 2 <= len(rows)-1 <= CSV_MAX_ROWS:
        raise ValueError("趋势数据需要2至200条有效记录")
    previous = None
    cleaned_rows = [CSV_FIELDS]
    for idx, cells in enumerate(rows[1:], 2):
        if len(cells) != 5:
            raise ValueError(f"第{idx}行不是5列")
        when = cells[0].strip()
        try:
            timestamp = datetime.fromisoformat(when)
        except ValueError as exc:
            raise ValueError(f"第{idx}行时间无效，使用YYYY-MM-DD HH:MM[:SS]") from exc
        if timestamp.tzinfo is not None:
            raise ValueError("时间字段请使用不带时区的本地时间")
        if previous is not None and timestamp <= previous:
            raise ValueError("CSV时间必须严格递增")
        previous = timestamp
        numeric = []
        for j in range(1, 5):
            value = cells[j].strip()
            if value == "":
                numeric.append("")
                continue
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(f"第{idx}行第{j+1}列不是有效数值") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"第{idx}行第{j+1}列必须为非负有限数值")
            numeric.append(format(number,".12g"))
        cleaned_rows.append((when, *numeric))
    out = io.StringIO()
    writer=csv.writer(out,lineterminator="\n")
    writer.writerows(cleaned_rows)
    return out.getvalue(),len(cleaned_rows)-1


def attachment_meta(name: str, data: bytes) -> dict[str, Any]:
    ext=name.rsplit(".",1)[-1].lower() if "." in name else ""
    if ext not in ("mat", "csv"):
        raise ValueError("目前只接受MAT振动数据与CSV运行趋势数据")
    if ext=="mat" and len(data) > MAT_MAX_BYTES:
        raise ValueError("MAT文件大于25MB")
    sha=hashlib.sha256(data).hexdigest()
    if ext=="csv":
        cleaned, n=validate_trend_csv(data)
        return {"kind":"csv","name":name,"sha256":sha,"csv":cleaned,"rows":n}
    return {"kind":"mat","name":name,"sha256":sha,"content":data,
            "source_fs_hz":source_rate_hint(name)}
