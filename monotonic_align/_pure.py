# -*- coding: utf-8 -*-
"""
monotonic_align 的纯 numpy 实现 —— 与 monotonic_align/core.pyx 数学等价。

为什么需要它
------------
本机的 Cython 扩展 core.cp38-win_amd64.pyd 在通过包导入时会报
    ImportError: DLL load failed while importing core: 参数错误。
即使用官方指定的 Cython 0.29.37 + 已安装的 MSVC/Windows SDK 重新编译，
现象依旧；而同一个 .pyd 复制到短路径下作为顶层模块导入却能成功。
属于 Windows 下扩展模块加载的环境怪象，难以根治。

由于 VITS 的推理路径（models.SynthesizerTrn.infer）**完全不会调用
maximum_path**，它只在训练时用；因此用纯 numpy 等价实现兜底即可让
训练/推理全部正常工作，且不影响结果正确性。

core.pyx 的原始实现（标量，逐元素循环）：

    cdef void maximum_path_each(int[:,::1] path, float[:,::1] value,
                                int t_y, int t_x, float max_neg_val=-1e9):
      cdef int index = t_x - 1
      for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
          if x == y:
            v_cur = max_neg_val
          else:
            v_cur = value[y-1, x]
          if x == 0:
            if y == 0:
              v_prev = 0.
            else:
              v_prev = max_neg_val
          else:
            v_prev = value[y-1, x-1]
          value[y, x] += max(v_prev, v_cur)

      for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
          index = index - 1

参数说明（注意本仓库的命名与注释是「反」的）：
  value / path 形状都是 [t_y, t_x]
  在本仓库中 t_y = 梅尔帧数，t_x = 文本音素数
  因此要求 t_x <= t_y，否则所有行区间为空、path 全零
  （这也是预处理阶段要剔除「音素数 > 梅尔帧数」样本的原因）
"""
import numpy as np

MAX_NEG_VAL = -1e9


def maximum_path_each_scalar(path, value, t_y, t_x, max_neg_val=MAX_NEG_VAL):
    """逐元素标量版：逐行照抄 core.pyx，作为向量化版的对拍参照。"""
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += v_prev if v_prev > v_cur else v_cur

    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


def maximum_path_each(path, value, t_y, t_x, max_neg_val=MAX_NEG_VAL):
    """向量化版：按 y 逐行推进，行内所有 x 一次性算完。

    value[y, x] 只依赖 value[y-1, :]（上一行），同一行内部无依赖，
    因此可以整行向量化。
    """
    t_y, t_x = int(t_y), int(t_x)
    if t_x <= 0 or t_y <= 0:
        return

    for y in range(t_y):
        lo = t_x + y - t_y
        if lo < 0:
            lo = 0
        hi = y + 1
        if hi > t_x:
            hi = t_x
        if lo >= hi:
            continue
        if y == 0:
            # y=0 时唯一可能的位置是 x=0：
            # v_cur = max_neg_val, v_prev = 0，max = 0，值不变，直接跳过
            continue

        xs = np.arange(lo, hi)
        prev = value[y - 1]

        v_cur = prev[xs].astype(np.float32, copy=True)
        v_cur[xs == y] = max_neg_val           # x == y 时不允许「原地停留」

        v_prev = np.full(xs.shape, max_neg_val, dtype=np.float32)
        nz = xs > 0
        if nz.any():
            v_prev[nz] = prev[xs[nz] - 1]

        value[y, xs] += np.maximum(v_prev, v_cur)

    # 反向回溯：t_y 次循环，开销可忽略
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1


def maximum_path_c(paths, values, t_ys, t_xs):
    """对应 core.pyx::maximum_path_c，逐样本循环。

      paths : int32   [b, t_y, t_x]
      values: float32 [b, t_y, t_x]
    """
    for i in range(paths.shape[0]):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
