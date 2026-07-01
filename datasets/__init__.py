"""Dataset utilities for Nexara."""

__all__ = ["StreamingTokenCacheDataset", "TokenBlockDataset", "TokenCacheDataset"]


def __getattr__(name: str):
    if name == "TokenBlockDataset":
        from .text import TokenBlockDataset

        return TokenBlockDataset
    if name == "TokenCacheDataset":
        from .token_cache import TokenCacheDataset

        return TokenCacheDataset
    if name == "StreamingTokenCacheDataset":
        from .token_cache import StreamingTokenCacheDataset

        return StreamingTokenCacheDataset
    raise AttributeError(name)
