# RedAlert2_Adjutant_TTS 

> 《红色警戒2 · 心灵终结》四阵营副官语音合成 
> 一个基于 **VITS 推理 + Gradio ** 的可视化语音生成服务

---

## 一、项目简介

**做什么**：输入一段文字 → 选择阵营（盟军 / 苏联 / 焚风 / 厄普西隆）与语言（英 / 中 / 日 / 混读）→ 生成对应副官声线的语音，结果可在网页里试听、也可落盘为 wav。

**技术底座**：VITS（[VITS-fast-fine-tuning](https://github.com/plachtaa/VITS-fast-fine-tuning)）+
CJE 三语底模（`uma_trilingual`，22050 Hz，`cjke_cleaners2` 文本前端），
在游戏内语音上微调出 4 个说话人，推理端用 Gradio 4.44 做本地网页界面。

| 阵营 | 说话人键名 | 说话人 ID |
| --- | --- | --- |
| 盟军 | `us_eva` | 3 |
| 苏联 | `su_eva` | 2 |
| 焚风 | `ff_eva` | 1 |
| 厄普西隆 | `ep_eva` | 0 |

> 说话人 ID **不是固定值**，由 `visual/modified_finetune_speaker.json` 的 `speakers` 字段决定。
> 代码里读配置，不要硬编码。

---

## 二、目录结构

```
RedAlert2_Adjutant_TTS/
├── README.md                          本文件
├── requirements.txt                   推理所需最小依赖（可直接 pip install）
├── LICENSE                            许可（继承上游）
│
├── visual/                            ★ 服务入口与模型配置
│   ├── app.py                         Gradio 网页服务主程序（唯一启动入口；深色 HUD 主题，样式内置于此文件）
│   ├── 启动服务.bat                    一键启动（GBK + CRLF，调用 ../venv）
│   └── outputs/                       生成音频落盘目录（含 history.tsv 索引）
│       └── .gitkeep
│
├── commons.py                         通用工具：intersperse / 卷积 padding / 初始化 / 切片
├── utils.py                           配置与权重加载（get_hparams_from_file / load_checkpoint）、tag_cjke 语言标注
├── models.py                          ★ 模型结构：SynthesizerTrn（推理调用 net_g.infer）
├── modules.py                         VITS 基础层：残差块 / WaveNet / 归一化流
├── attentions.py                      多头注意力（Transformer 编码器用）
├── transforms.py                      单调有理二次样条（归一化流用）
│
├── text/                              多语言文本前端（文字 → 音素符号序列）
│   ├── __init__.py                    text_to_sequence / _clean_text 入口
│   ├── cleaners.py                    ★ 清洗器集合（本项目用 cjke_cleaners2）
│   ├── symbols.py                     音素符号表定义
│   ├── mandarin.py                    中文 → IPA（pypinyin / jieba / cn2an）
│   ├── english.py                     英文 → IPA（inflect / eng_to_ipa / unidecode）
│   ├── japanese.py                    日文 → IPA（pyopenjtalk / unidecode）
│   ├── korean.py                      韩文 → IPA（jamo / ko_pron）
│   ├── sanskrit.py                    梵文转写（cleaners.py 顶层 import，必须存在）
│   ├── thai.py                        泰文数字/拉丁转写（同上，必须存在）
│   └── LICENSE                        上游文本前端许可
│
└── monotonic_align/                   单调对齐（训练用；推理仅需可 import）
    ├── __init__.py                    ★ 已改：优先编译版，失败自动退回纯 numpy
    ├── _pure.py                       纯 numpy 实现（current 默认后端）
    ├── core.pyx                       上游 Cython 源码（可选编译）
    └── setup.py                       编译脚本（可选）
```

该项目所使用的模型权重开源在hugging face：

https://huggingface.co/gurentan/RedAlert2_Adjutant_Voice_Model

可以在该仓库下载模型权重 `G_latest.pth` 和配置文件 `modified_finetune_speaker.json` ，放入项目中的 `visual` 文件夹中。

**放模型权重后的最终形态**：

```
RedAlert2_Adjutant_TTS/
└── visual/
    ├── app.py
    ├── modified_finetune_speaker.json
    └── G_latest.pth        
```

---

## 三、运行环境

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| 系统 | Windows 10 / 11 | 代码跨平台，但一键脚本是 `.bat` |
| Python | **3.8.x** | 上游 `Cython 0.29.x` / `opencc 1.1.1` 在 3.8 上验证；更高版本未验证 |
| 显卡 | NVIDIA，显存 **≥ 6 GB** | 实测推理占用约 3.0 GB；无卡可 `--device cpu`，约慢 10 倍 |
| 磁盘 | ≥ 6 GB | CUDA 版 torch 约 2.4 GB + 模型 152 MB + 依赖 |
| 网络 | 能访问 PyPI / PyTorch 源 | 国内建议用镜像 |

参考实测机器：RTX 3060 Laptop 6 GB + i7-12700H + 16 GB，合成一句 2 秒语音约 1～2 秒。

---

## 四、安装步骤

以下命令均在 **`RedAlert2_Adjutant_TTS/` 根目录**执行（Windows 命令行 / PowerShell）。

### 步骤 1 — 创建虚拟环境

```bat
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
```

### 步骤 2 — 先单独装 CUDA 版 torch

> ⚠️ 必须单独装：PyPI / 国内镜像站上的 Windows `torch` 是 **CPU 版**，
> 装错会静默退化成 CPU 推理，速度差 10 倍且不报错。

```bat
:: 官方源
venv\Scripts\pip.exe install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

:: 国内镜像
venv\Scripts\pip.exe install torch==2.1.2 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
```

验证：

```bat
venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available())"
:: 期望输出类似 2.1.2+cu121 True
```

### 步骤 3 — 安装其余依赖

```bat
venv\Scripts\pip.exe install -r requirements.txt
```

### 步骤 4 — 放入模型权重

见第二节。权重放到 `visual/G_latest.pth`。

### 步骤 5 — 启动

双击 `visual\启动服务.bat`，或：

```bat
venv\Scripts\python.exe visual\app.py
```

浏览器自动打开 `http://127.0.0.1:7860`。启动成功的日志形如：

```
正在加载模型 ...
  模型：...\visual\G_latest.pth
  配置：...\visual\modified_finetune_speaker.json
  设备：cuda:0   采样率：22050 Hz
  说话人：{'ep_eva': 0, 'ff_eva': 1, 'su_eva': 2, 'us_eva': 3}
正在启动服务，地址：http://127.0.0.1:7860
Running on local URL:  http://127.0.0.1:7860
```

### 启动参数

```bat
venv\Scripts\python.exe visual\app.py --port 7861       :: 换端口
venv\Scripts\python.exe visual\app.py --host 0.0.0.0    :: 局域网可访问
venv\Scripts\python.exe visual\app.py --device cpu      :: 强制 CPU
venv\Scripts\python.exe visual\app.py --share           :: Gradio 公网临时链接
venv\Scripts\python.exe visual\app.py --no-browser      :: 不自动开浏览器
venv\Scripts\python.exe visual\app.py --model <pth> --config <json>
```

> 7860 被占用时会自动改用 7861、7862……，实际地址以窗口输出为准。

---

## 五、使用方式

### 5.1 网页

网页为《红色警戒2 · 心灵终结》战术指挥终端风格：金属切角面板、战术网格、扫描雷达、声纹监视器和通讯记录表共同构成主界面。主题、SVG 和交互脚本全部内联在 `app.py`，不加载外部字体、图片或 CDN；页面美化不改变推理流程，也不影响 `/tts` 接口。

**页面布局**

1. **顶部指挥区**：显示 `BATTLE CONTROL ONLINE`、`MENTAL OMEGA` 与当前服务状态，右侧为持续扫描的战术雷达，底部标注 `SECTOR SCAN`。
2. **链路状态栏**：展示连接状态、四个阵营声线、支持语言和本地推理节点。
3. **01 任务指令**：输入待朗读文本，选择阵营与语言，调整语速及高级参数，并决定是否保存音频。
4. **02 语音回传**：生成前显示待命声纹，生成期间切换为传输动画；下方依次为音频播放器、状态信息和实际标注文本。
5. **点击载入示例**：点击任意一行战术台词，可一次性填入文字、阵营、语言、语速与高级参数，再按需修改。

选择目标阵营后，页面强调色会同步切换：盟军为战术蓝、苏联为警戒红、焚风为青绿、厄普西隆为心灵紫。

| 区域 / 控件 | 说明 |
| --- | --- |
| 要朗读的文字 | 任意文本；输入框右下方实时显示字符数，Mix 模式可手写 `[ZH]/[JA]/[EN]` 标记 |
| 目标阵营 | 盟军 / 苏联 / 焚风 / 厄普西隆；同时决定页面强调色 |
| 语言 | English / 简体中文 / 日本語 / Mix；说明文字位于选择框下方 |
| 语音倍速 | 0.1～2.0（映射为 `length_scale = 1/speed`） |
| 感情变化程度 | 位于「高级参数」中；`noise_scale`，默认 0.667，> 1.0 可能发散 |
| 音素时长随机度 | 位于「高级参数」中；`noise_scale_w`，默认 0.6 |
| 把生成的音频保存到 visual/outputs/ | 勾选后写入 wav，并向 `history.tsv` 追加一条索引 |
| 建立语音链路 // GENERATE | 提交生成任务；生成期间显示传输状态并锁定按钮，完成后自动恢复 |
| 生成结果 | 可直接试听；未勾选保存时仅保留在临时目录 |
| 状态 | 成功时显示阵营、语速、时长及保存路径，失败时显示具体原因 |
| 实际送入模型的标注文本 | 用来核对语言标记是否正确 |

**操作流程**

1. 输入文字，或从「点击载入示例」中选择一条预设台词。
2. 选择目标阵营与语言；通常保持默认语速即可，需要时展开「高级参数」调整感情和节奏。
3. 点击「建立语音链路 // GENERATE」，或直接在文本框中按回车。
4. 等待右侧声纹从 `TRANSMITTING` 恢复为 `UPLINK READY`，随后试听并核对状态与标注文本。

单一语言选择时程序自动整段包裹：`[EN]...[EN]` / `[ZH]...[ZH]` / `[JA]...[JA]`。
Mix 模式下：文本里已有任意标记 → 原样使用；没有 → 调 `utils.tag_cjke()` 自动识别。

页面采用响应式布局：宽屏为左右双栏，窄屏自动上下堆叠；移动端示例表格可独立横向滑动。若系统启用「减少动态效果」，雷达、声纹和入场动画会自动降级。

### 5.2 HTTP / Python 调用

```python
from gradio_client import Client
c = Client("http://127.0.0.1:7860")
audio, status, tagged = c.predict(
    "Mission accomplished.", "焚风", "English（英文）", 1.0, 0.667, 0.6, True,
    api_name="/tts",          # 必须用 api_name，不要用 fn_index
)
```

---

## 六、常见问题

**Q1：双击 `.bat` 后窗口一闪就退，或提示"服务已退出"？**
几乎都是 7860 端口被占用（Gradio 4.44 端口冲突会静默退出）。`app.py` 已内置端口探测，会自动换 7861+；若仍失败，查占用：
`netstat -ano | findstr :7860`（最后一列是 PID）。

**Q2：日志刷屏几千行 DEBUG？**
上游 `utils.py` 顶层有 `logging.basicConfig(level=logging.DEBUG)`，`app.py` 已把它压回 `WARNING`。若你自己写了脚本直接 `import utils`，需要自行压制日志级别。

**Q3：报 `DLL load failed while importing core`？**
`monotonic_align/__init__.py` 会自动退回纯 numpy 实现 `_pure.py`，**不影响合成结果**，无需处理。想彻底消除可编译：
`cd monotonic_align && python setup.py build_ext --inplace`（需 `Cython==0.29.21`）。

**Q4：报 `ModuleNotFoundError: No module named 'regex'`？**
原仓库 `requirements.txt` 漏写，本包已补。装 `pip install regex` 即可。

**Q5：`import text` 时报 `indic_transliteration` / `num_thai` 缺失？**
这两个是 `cleaners.py` 顶层 import 的（对应 Sanskrit / Thai），即使不用也必须装。

**Q6：第一次合成日文时报 `URLError: CERTIFICATE_VERIFY_FAILED` 或卡住很久？**
`pyopenjtalk` 首次 `import` 会去 GitHub 拉 Open JTalk 词典（约 30 MB）。网络受限或系统证书链不全时就会失败（本机实测 TLS 握手证书校验失败）。解决：换能用 `certifi` 证书的 Python / 设置代理后重试，或手动下载词典解压到 `site-packages/pyopenjtalk/` 下（目录名以该包内的 `OPEN_JTALK_DICT_DIR` 为准，上游文档 docs/04 有完整步骤）。**注意：`text/japanese.py` 顶层就 import 了它，所以即使只合成中/英文，它也必须能 import 成功。**

**Q7：合成出来是"哒哒哒"电流声或静音？**
优先核对 `visual/modified_finetune_speaker.json` 与 `G_latest.pth` 是否配套；其次确认语言标记与文本是否匹配（中文字符串选了 English 会清洗成空序列，程序会报"文本清洗后为空"）。

---

## 七、许可与免责

- 训练框架：[plachtaa/VITS-fast-fine-tuning](https://github.com/plachtaa/VITS-fast-fine-tuning)
- 单调对齐：[jaywalnut310/vits](https://github.com/jaywalnut310/vits)（`core.pyx` 原样保留）
- 语音素材与台词版权归 **《命令与征服：红色警戒2》及其模组《心灵终结（Mental Omega）》** 原作者所有
- 本包**不包含**任何游戏原始音频素材，仅供技术学习与个人研究；请勿将生成音频用于商业用途或公开传播

具体条款见 `LICENSE`。
