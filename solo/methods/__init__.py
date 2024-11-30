from solo.methods.base import BaseMethod
from solo.methods.supcon_backbone import SupConBackbone
from solo.methods.supcon_disentangle import SupConDisentangle
from solo.methods.mocov2 import MoCoV2Plus
from solo.methods.mocov2_disentangle import MocoDisentangle


METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "supconbackbone": SupConBackbone,
    "supcondisentangle": SupConDisentangle,
    "mocov2":MoCoV2Plus,
    "mocodisentangle": MocoDisentangle,
}
__all__ = [
    "BaseMethod",
    "SupConBackbone",
    "SupConDisentangle",
    "MoCoV2Plus",
    "MocoDisentangle",
]
