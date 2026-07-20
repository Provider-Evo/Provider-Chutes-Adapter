

from typing import Any, Dict, List, Optional

# 流式请求最大 token 数
MAX_TOKENS_STREAMING: int = 65536
# 非流式请求最大 token 数
MAX_TOKENS_NON_STREAMING: int = 4096


def build_payload(
    messages: List[Dict[str, Any]],
    model: str,
    stream: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop: Optional[Any] = None,
) -> Dict[str, Any]:
    """构建聊天请求体。

    Args:
        messages: 消息列表。
        model: 模型名称。
        stream: 是否启用流式响应。
        max_tokens: 最大生成 token 数，None 时按流式/非流式默认值。
        temperature: 采样温度。
        top_p: 核采样概率。
        stop: 停止词。

    Returns:
        请求体字典。
    """
    if max_tokens is None:
        max_tokens = MAX_TOKENS_STREAMING if stream else MAX_TOKENS_NON_STREAMING

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if stop:
        payload["stop"] = stop
    return payload
