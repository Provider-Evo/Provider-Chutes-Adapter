"""Chutes 客户端——每个 API Key 一个候选项"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp

from src.core.dispatch.candidate import Candidate, make_id
from src.foundation.logger import get_logger

from .constants import BASE_URL, CHAT_PATH, CAPS
from .headers import build_headers
from .payloads import build_payload
from .sse import parse_sse_line

logger = get_logger(__name__)

MAX_RETRIES: int = 3


# 连续失败阈值，超过后进入冷却
_FAILURE_THRESHOLD: int = 3
# Key 失效冷却时间（秒）
_RECOVERY_INTERVAL: float = 60.0
# 鉴权失败状态码——此类 Key 直接标记为无效
_AUTH_ERROR_CODES = frozenset({401, 402, 403})


class _KeyState:
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
        if self.consecutive_failures >= _FAILURE_THRESHOLD:
            if time.monotonic() - self.last_error_time < _RECOVERY_INTERVAL:
                return False
        return True

    def try_recover(self) -> None:
        """尝试从冷却状态中恢复。

        若冷却时间已过，重置失败计数并恢复可用状态。
        """
        if not self._valid:
            if time.monotonic() - self.last_error_time >= _RECOVERY_INTERVAL:
                self._valid = True
                self.consecutive_failures = 0
                logger.info("chutes Key 已从无效状态恢复: %s...", self.key[:16])
        elif self.consecutive_failures >= _FAILURE_THRESHOLD:
            if time.monotonic() - self.last_error_time >= _RECOVERY_INTERVAL:
                self.consecutive_failures = 0
                logger.info("chutes Key 冷却结束，恢复可用: %s...", self.key[:16])

    def mark_success(self) -> None:
        """标记本次请求成功，重置失败计数。
        """
        self.consecutive_failures = 0

    def mark_failure(self, status: int = 0) -> None:
        """标记本次请求失败并更新状态。

        Args:
            status: HTTP 响应状态码，0 表示网络异常。
        """
        self.last_error_time = time.monotonic()
        if status in _AUTH_ERROR_CODES:
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


class ChutesClient:
    """Chutes HTTP 客户端。

    每个 API Key 对应一个候选项，统一由 TAS 算法调度。
    不使用 asyncio.Lock，依赖事件循环单线程特性保证并发安全。
    """

    def __init__(self) -> None:
        """初始化客户端实例。"""
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: List[str] = []
        self._candidates: List[Candidate] = []
        self._key_states: List[_KeyState] = []

    async def init_immediate(self, session: aiohttp.ClientSession) -> None:
        """立即初始化，不阻塞。

        注入共享会话，构建初始 Key 状态列表和候选项列表。

        Args:
            session: 共享的 aiohttp 会话。
        """
        self._session = session
        from ..accounts import API_KEYS

        self._key_states = [
            _KeyState(k) for k in API_KEYS if isinstance(k, str) and k.strip()
        ]
        self._rebuild_candidates()
        logger.info(
            "chutes 客户端初始化完成，API Key 数量: %d",
            len(self._key_states),
        )

    async def background_setup(self) -> None:
        """后台完善操作。

        Chutes 平台为 API Key 鉴权，无需登录，此方法为空。
        """
        return

    def update_models(self, models: List[str]) -> None:
        """更新模型列表，同步刷新所有候选项的 models 字段。

        Args:
            models: 新的模型列表。
        """
        self._models = list(models)
        for cand in self._candidates:
            cand.models = list(models)

    def _rebuild_candidates(self) -> None:
        """根据当前 Key 状态重建候选项列表。
        """
        self._candidates = [
            Candidate(
                id=make_id("chutes", ks.key[:16]),
                platform="chutes",
                resource_id=ks.key[:16],
                models=list(self._models),
                context_length=None,
                meta={"api_key": ks.key},
                **CAPS,
            )
            for ks in self._key_states
        ]

    async def candidates(self) -> List[Candidate]:
        """返回当前可用候选项列表。

        先尝试恢复冷却中的 Key，再筛选可用项。

        Returns:
            可用候选项列表。
        """
        available: List[Candidate] = []
        for ks, cand in zip(self._key_states, self._candidates):
            ks.try_recover()
            if ks.is_available():
                available.append(cand)
        return available

    async def ensure_candidates(self, count: int) -> int:
        """返回可用候选项数量。

        Args:
            count: 期望的候选项数量（本实现忽略此参数）。

        Returns:
            实际可用候选项数量。
        """
        available = 0
        for ks in self._key_states:
            ks.try_recover()
            if ks.is_available():
                available += 1
        return available

    def _find_key_state(self, candidate: Candidate) -> Optional[_KeyState]:
        """根据候选项找到对应的 Key 状态。

        Args:
            candidate: 候选项实例。

        Returns:
            对应的 _KeyState，未找到返回 None。
        """
        api_key = candidate.meta.get("api_key", "")
        for ks in self._key_states:
            if ks.key == api_key:
                return ks
        return None

    async def complete(
        self,
        candidate: Candidate,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        *,
        thinking: bool = False,
        search: bool = False,
        **kw: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """执行聊天补全，含指数退避重试。

        Args:
            candidate: 候选项。
            messages: 消息列表。
            model: 模型名。
            stream: 是否流式。
            thinking: 是否启用推理（透传至 payload）。
            search: 是否启用搜索（透传至 payload）。
            **kw: 额外关键字参数（max_tokens、temperature、top_p、stop）。

        Yields:
            文本片段或元数据字典。

        Raises:
            Exception: 重试耗尽后抛出最后一次异常。
        """
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                delay = 1.0 * (2 ** (attempt - 1))
                logger.warning(
                    "chutes 重试 %d/%d，等待 %.1fs",
                    attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            try:
                async for chunk in self._do_request(
                    candidate, messages, model, stream, **kw
                ):
                    yield chunk
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("chutes 请求失败 %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
        if last_exc is not None:
            raise last_exc

    async def _do_request(
        self,
        candidate: Candidate,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kw: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """执行单次 HTTP 请求。

        Args:
            candidate: 候选项。
            messages: 消息列表。
            model: 模型名。
            stream: 是否流式。
            **kw: 额外参数（max_tokens、temperature、top_p、stop）。

        Yields:
            文本片段或元数据字典。

        Raises:
            Exception: HTTP 状态码非 200 或请求异常时抛出。
        """
        ks = self._find_key_state(candidate)
        if ks is None:
            raise Exception("chutes: 未找到对应 API Key 状态")

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

        try:
            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(connect=10, total=600),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    ks.mark_failure(resp.status)
                    raise Exception(
                        "chutes HTTP {}: {}".format(resp.status, body[:300])
                    )

                if stream:
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
                    ks.mark_success()
                else:
                    obj = await resp.json()
                    ks.mark_success()
                    choices = obj.get("choices") or []
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            yield content
                    usage = obj.get("usage")
                    if usage:
                        yield {"usage": usage}
        except Exception as exc:
            if ks is not None:
                ks.mark_failure(0)
            raise exc

    async def close(self) -> None:
        """清理资源，session 由外部管理，此处不关闭。
        """
        return
