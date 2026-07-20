from __future__ import annotations

"""Chutes HTTP 客户端辅助模块。

职责：
    承载 :class:`_KeyState`（单Key运行时状态）与请求/响应级别的纯函数
    （构造请求、解析非流式/流式响应），供 ``client.py`` 中的
    :class:`ChutesClient` facade 调用。拆分自 ``client.py``，不改变任何
    现有行为。
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

import aiohttp

from src.core.dispatch.cand import Candidate
from src.foundation.logger import get_logger

from ..consts import BASE_URL, CHAT_PATH
from ..headers import build_headers
from ..payload import build_payload
from .sse import parse_sse_line

logger = get_logger(__name__)

# 连续失败阈值，超过后进入冷却
FAILURE_THRESHOLD: int = 3
# Key 失效冷却时间（秒）
RECOVERY_INTERVAL: float = 60.0
# 鉴权失败状态码——此类 Key 直接标记为无效
AUTH_ERROR_CODES = frozenset({401, 402, 403})


class KeyState:
    """单个 API Key 的运行时状态。

    不使用锁——依赖 asyncio 单线程事件循环保证操作原子性。
    """

    __slots__ = ("key", "_valid", "consecutive_failures", "last_error_time")

    def __init__(self, key: str) -> None:
        """初始化 Key 状态。

        Args:
            key: API Key 字符串。
        """
        self.key: str = key
        self._valid: bool = True
        self.consecutive_failures: int = 0
        self.last_error_time: float = 0.0

    def is_available(self) -> bool:
        """判断当前 Key 是否可用。

        副作用分离：不在此方法内修改状态，由 try_recover() 负责恢复。

        Returns:
            True 表示可用，False 表示不可用。
        """
        if not self._valid:
            return False
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            if time.monotonic() - self.last_error_time < RECOVERY_INTERVAL:
                return False
        return True

    def try_recover(self) -> None:
        """尝试从冷却状态中恢复。

        若冷却时间已过，重置失败计数并恢复可用状态。
        """
        if not self._valid:
            if time.monotonic() - self.last_error_time >= RECOVERY_INTERVAL:
                self._valid = True
                self.consecutive_failures = 0
                logger.info("chutes Key 已从无效状态恢复: %s...", self.key[:16])
        elif self.consecutive_failures >= FAILURE_THRESHOLD:
            if time.monotonic() - self.last_error_time >= RECOVERY_INTERVAL:
                self.consecutive_failures = 0
                logger.info("chutes Key 冷却结束，恢复可用: %s...", self.key[:16])

    def mark_success(self) -> None:
        """标记本次请求成功，重置失败计数。"""
        self.consecutive_failures = 0

    def mark_failure(self, status: int = 0) -> None:
        """标记本次请求失败并更新状态。

        Args:
            status: HTTP 响应状态码，0 表示网络异常。
        """
        self.last_error_time = time.monotonic()
        if status in AUTH_ERROR_CODES:
            self._valid = False
            logger.warning(
                "chutes Key 鉴权失败 (HTTP %d)，已标记无效: %s...",
                status, self.key[:16],
            )
        else:
            self.consecutive_failures += 1
            logger.warning(
                "chutes Key 连续失败 %d 次: %s...",
                self.consecutive_failures, self.key[:16],
            )


async def stream_chat_response(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """解析流式 SSE 响应。

    Args:
        resp: HTTP 响应对象。

    Yields:
        文本片段或元数据字典。
    """
    async for line in resp.content:
        if not line:
            continue
        text = line.decode("utf-8", errors="replace").strip()
        if not text or not text.startswith("data:"):
            continue
        data_str = text[5:].strip()
        if data_str == "[DONE]":
            break
        parsed = parse_sse_line(data_str)
        if parsed is not None:
            yield parsed


async def nonstream_chat_response(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """解析非流式 JSON 响应。

    Args:
        resp: HTTP 响应对象。

    Yields:
        文本内容或含 usage 的字典。
    """
    obj = await resp.json()
    choices = obj.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if content:
            yield content
    usage = obj.get("usage")
    if usage:
        yield {"usage": usage}


def build_chat_request(
    candidate: Candidate,
    messages: List[Dict[str, Any]],
    model: str,
    stream: bool,
    **kw: Any,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """构造 chutes 聊天补全请求的 URL/headers/payload。

    Args:
        candidate: 候选项。
        messages: 消息列表。
        model: 模型名。
        stream: 是否流式。
        **kw: 额外参数（max_tokens、temperature、top_p、stop）。

    Returns:
        (url, headers, payload) 三元组。
    """
    api_key = candidate.meta.get("api_key", "")
    headers = build_headers(api_key)
    payload = build_payload(
        messages=messages,
        model=model,
        stream=stream,
        max_tokens=kw.get("max_tokens"),
        temperature=kw.get("temperature"),
        top_p=kw.get("top_p"),
        stop=kw.get("stop"),
    )
    url = "{}{}".format(BASE_URL, CHAT_PATH)
    return url, headers, payload


async def dispatch_chat_response(
    resp: aiohttp.ClientResponse,
    stream: bool,
    ks: KeyState,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """校验响应状态码并转发至流式/非流式解析器。

    Args:
        resp: HTTP 响应对象。
        stream: 是否流式。
        ks: 对应的 Key 状态，非 200 时用于标记失败。

    Yields:
        文本片段或元数据字典。

    Raises:
        Exception: HTTP 状态码非 200 时抛出。
    """
    if resp.status != 200:
        body = await resp.text()
        ks.mark_failure(resp.status)
        raise Exception("chutes HTTP {}: {}".format(resp.status, body[:300]))

    if stream:
        async for chunk in stream_chat_response(resp):
            yield chunk
    else:
        async for chunk in nonstream_chat_response(resp):
            yield chunk
