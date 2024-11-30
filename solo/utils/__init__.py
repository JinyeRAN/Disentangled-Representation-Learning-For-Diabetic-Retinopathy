from solo.utils import (
    checkpointer,
    misc,
    test_utils
)

__all__ = [
    "checkpointer",
    "misc",
    "test_utils"
]


try:
    from solo.utils import auto_umap  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("auto_umap")
