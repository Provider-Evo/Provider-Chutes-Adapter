


from __future__ import annotations

from typing import Any, Dict, Optional, Union

from provider_sdk.extensions.platform.sse_common import parse_openai_sse_line

__all__ = ["parse_sse_line"]


def parse_sse_line(data_str: str) -> Optional[Union[str, Dict[str, Any]]]:
    """解析 SSE data 字段内容（OpenAI 兼容格式）。

    Args:
        data_str: ``data:`` 前缀之后的字符串，已去除前缀和首尾空白。

    Returns:
        文本片段（str）、元数据字典（dict）或 None（跳过该事件）。

    Raises:
        ValueError: SSE 事件包含 error 字段时抛出。
    """
    return parse_openai_sse_line(data_str)
