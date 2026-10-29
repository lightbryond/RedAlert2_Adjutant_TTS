import numpy as np
import torch

# ---------------------------------------------------------------------------
# monotonic_align 入口
#
# 优先使用编译好的 Cython 扩展；若加载失败（本机存在
# "DLL load failed while importing core" 的环境问题），自动退回纯 numpy 实现。
#
# VITS 的推理路径不调用 maximum_path（仅训练时使用），因此兜底方案对
# 文本转语音完全没有影响；训练时的性能差异也只有每步约 15ms。
#
# 原文件备份：同目录 __init__.py.orig
# ---------------------------------------------------------------------------
_BACKEND = None
maximum_path_c = None

for _dotted in (".monotonic_align.core", ".core"):
    try:
        _mod = __import__(__name__ + _dotted, fromlist=["maximum_path_c"])
        maximum_path_c = _mod.maximum_path_c
        _BACKEND = "cython" + _dotted
        break
    except Exception:
        continue

if maximum_path_c is None:
    from ._pure import maximum_path_c
    _BACKEND = "pure-numpy"


def maximum_path(neg_cent, mask):
  """ Cython 兼容接口
  neg_cent: [b, t_y, t_x]   (本仓库里 t_y = 梅尔帧数, t_x = 音素数)
  mask:     [b, t_y, t_x]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)
