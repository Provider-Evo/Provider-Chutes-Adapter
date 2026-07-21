from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp

from provider_sdk.model_ids import ModelIdRegistry

from src.core.dispatch.cand import Candidate, make_id
from src.foundation.config.reader import load_plugin_api_keys
from src.foundation.logger import get_logger

from .helpers.client_helpers import KeyState as _KeyState, build_chat_request, dispatch_chat_response
from .consts import CAPS, MODELS

_PLUGIN_DIR = Path(__file__).resolve().parents[2]

logger = get_logger(__name__)

MAX_RETRIES: int = 3


class ChutesClient:
    """Chutes HTTP 客户端。

    每个 API Key 对应一个候选项，统一由 TAS 算法调度。
    不使用 asyncio.Lock，依赖事件循环单线程特性保证并发安全。
    Key 状态与请求/响应构造拆分至 ``client_helpers.py``。
    """

    def __init__(self) -> None:
        """初始化客户端实例。"""
        self._session: Optional[aiohttp.ClientSession] = None
        self._model_registry = ModelIdRegistry("chutes")
        self._model_registry.load()
        self._models: List[str] = self._model_registry.merge_fallback(MODELS)
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
            _KeyState(k) for k in load_plugin_api_keys(_PLUGIN_DIR, API_KEYS)
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
        self._models = self._model_registry.register_many(models)
        for cand in self._candidates:
            cand.models = list(self._models)

    def _rebuild_candidates(self) -> None:
        """根据当前 Key 状态重建候选项列表。"""
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
        model = self._model_registry.resolve_upstream(model)
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

        url, headers, payload = build_chat_request(
            candidate, messages, model, stream, **kw
        )

        try:
            async with self._session.post(
                url,
                json=payload,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(connect=10, total=600),
            ) as resp:
                async for chunk in dispatch_chat_response(resp, stream, ks):
                    yield chunk
                ks.mark_success()
        except Exception as exc:
            if ks is not None:
                ks.mark_failure(0)
            raise exc

    async def close(self) -> None:
        """清理资源，session 由外部管理，此处不关闭。"""
        return
