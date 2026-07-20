

from typing import Dict


def build_headers(api_key: str) -> Dict[str, str]:
    """构建 Chutes 接口请求头。

    Args:
        api_key: Chutes API Key。

    Returns:
        完整请求头字典。
    """
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
