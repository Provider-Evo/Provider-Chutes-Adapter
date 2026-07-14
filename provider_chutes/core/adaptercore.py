"""
adaptercore 模块。

本文件为 Provider-Evo 项目标准模块，使用以下约定：

- 模块路径：provider-plugin.Provider-Chutes-Adapter.provider_chutes.core.adaptercore
- 文件名：adaptercore.py
- 父包：provider-plugin/Provider-Chutes-Adapter/provider_chutes/core

职责：

    作为 SDK 兼容入口，转发到 ``provider_*.core`` 下的真实实现层。
    此模式让 ``from provider_xxx import adapter`` 与 ``from provider_xxx.adapter import …``
    同时可用，无需调用方关心内部布局。

对外接口：

    本模块的 ``__all__`` 列出对外可导入的符号集合；其他内部符号
    可能在重构中调整，调用方应只依赖 ``__all__`` 暴露的稳定 API。

集成：

    - SDK 入口：``plugin.py`` 中 ``create_plugin()`` 引用本模块以构造 platform adapter。
    - 入口路由：``provider-self/src/routes/openai`` 通过 ``from src.core...`` 间接使用。
    - 测试：本目录下的 ``tests/`` 子目录覆盖本模块的核心逻辑。

依赖：

    - 仅依赖 ``provider-sdk`` 与 Python 3.8+ 标准库；不引入第三方 HTTP 库。
    - 不直接读环境变量；所有配置走 ``config/main_config.toml``。

修改指引：

    - 调整本模块时同步更新 ``docs-src/plugins/<name>.md`` 与对应 ``tests/``。
    - 保持单文件 200-400 行；超长请拆为子包并通过 ``__init__.py`` 重新导出。
    - 严禁放置 placeholder / 兜底 / 伪装通过的代码（见 ``AGENTS.md`` Hard Constraints）。
"""


from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp

from src.core.dispatch.candidate import Candidate
from src.core.utils.compat.models_cache import ModelsCache
from src.foundation.logger import get_logger
from provider_sdk.extensions.platform.adapter import PlatformAdapter
from .constants import CAPS, FETCH_MODELS_ENABLED, MODEL_FETCH_INTERVAL, MODELS

logger = get_logger(__name__)

__all__ = ["Adapter", "ChutesAdapter"]


class Adapter(PlatformAdapter):
    """Chutes 平台适配器。

    通过 Chutes AI API 访问大模型。
    每个 API Key 作为一个独立候选项，由 Provider-V2 核心 TAS 算法调度。
    """

    def __init__(self) -> None:
        """初始化适配器实例。"""
        self._client: Any = None
        self._models: List[str] = list(MODELS)
        self._cache: Optional[ModelsCache] = None
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        """返回平台标识名。

        Returns:
            平台小写标识。
        """
        return "chutes"

    @property
    def supported_models(self) -> List[str]:
        """返回当前支持的模型列表。

        Returns:
            模型名称列表。
        """
        return list(self._models)

    @property
    def default_capabilities(self) -> Dict[str, bool]:
        """返回平台默认能力字典。

        Returns:
            能力布尔字典。
        """
        return CAPS

    async def init(self, session: aiohttp.ClientSession) -> None:
        """初始化适配器——立即注册，后台完善。

        先以硬编码数据注册到 registry，后台 Task 执行耗时操作。

        Args:
            session: 共享的 aiohttp 会话。
        """
        from .client import ChutesClient  # noqa: PLC0415

        self._client = ChutesClient()
        await self._client.init_immediate(session)

        self._cache = ModelsCache(
            platform="chutes",
            fallback_models=MODELS,
            fetch_enabled=FETCH_MODELS_ENABLED,
        )
        cached = await self._cache.load()
        if cached:
            self._models = cached
            self._client.update_models(self._models)

        self._refresh_task = asyncio.ensure_future(self._background_init())

    async def _background_init(self) -> None:
        """后台初始化：完成耗时操作后持续刷新。
        """
        try:
            await self._client.background_setup()
        except Exception as exc:
            logger.warning("chutes 后台初始化失败: %s", exc)

        if self._cache is not None:
            asyncio.ensure_future(
                self._cache.start_refresh_loop(
                    fetch_fn=self.fetch_remote_models,
                    interval=MODEL_FETCH_INTERVAL,
                    on_update=self._on_models_updated,
                )
            )

    async def _on_models_updated(self, models: List[str]) -> None:
        """模型列表更新回调。

        Args:
            models: 新的模型列表。
        """
        self._models = models
        if self._client is not None:
            self._client.update_models(models)
        logger.info("chutes 模型列表已更新: %d 个", len(models))

    async def fetch_remote_models(self) -> List[str]:
        """拉取远程模型列表。

        Chutes 平台模型列表固定，直接返回硬编码列表。

        Returns:
            模型名称列表。
        """
        return list(MODELS)

    async def candidates(self) -> List[Candidate]:
        """返回当前可用候选项。

        Returns:
            候选项列表。
        """
        if self._client is None:
            return []
        return await self._client.candidates()

    async def ensure_candidates(self, count: int) -> int:
        """确保候选项数量。

        Args:
            count: 期望的候选项数量。

        Returns:
            实际可用候选项数量。
        """
        if self._client is None:
            return 0
        return await self._client.ensure_candidates(count)

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
        """执行聊天补全，委托给 client。

        Args:
            candidate: 候选项。
            messages: 消息列表。
            model: 模型名。
            stream: 是否流式。
            thinking: 是否启用推理。
            search: 是否启用搜索。
            **kw: 额外关键字参数。

        Yields:
            文本片段或元数据字典。
        """
        async for chunk in self._client.complete(
            candidate, messages, model, stream,
            thinking=thinking, search=search, **kw,
        ):
            yield chunk

    async def close(self) -> None:
        """关闭适配器，释放资源。
        """
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                logger.debug("chutes 后台刷新任务已取消")
        if self._client is not None:
            await self._client.close()


ChutesAdapter = Adapter

