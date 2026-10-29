# -*- coding: utf-8 -*-
"""
心灵终结 · 四阵营副官语音合成服务（Gradio 网页端）
===========================================================================
基于微调后的 VITS 模型，在本地启动一个网页服务：
  * 输入任意文字
  * 选择阵营（盟军 / 苏联 / 焚风 / 厄普西隆）
  * 调整语音倍速（0.1 ~ 2.0）
  * 选择语言（英文 / 中文 / 日语 / Mix 多语言混合）

语言处理方式（与官方 app 页面一致）
-----------------------------------
* 选择单一语言时：只需选语言，程序自动用对应的语言标记把整段文字包起来
    英文 -> [EN]... [EN]     中文 -> [ZH]...[ZH]     日语 -> [JA]...[JA]
* 选择 Mix 模式时：
    - 若你已经在句子里写好了 [ZH] / [JA] / [EN] 标记，则原样使用（官方做法）
    - 若没有写标记，程序会调用仓库自带的 tag_cjke() 自动逐字识别语言并补上标记
  两种方式可以混用，「实际送入模型的标注文本」框中会显示最终结果。

启动
----
    python visual/app.py
    浏览器会自动打开 http://127.0.0.1:7860

常用参数
--------
    --port 7860        监听端口
    --host 127.0.0.1   监听地址（填 0.0.0.0 可供局域网访问）
    --share            生成 Gradio 公网临时链接
    --device cpu       强制用 CPU 推理
"""
import argparse
import datetime
import io
import os
import re
import sys
import contextlib

# ---------------------------------------------------------------------------
# 路径：本文件位于 <仓库根>/visual/ 下，模型与配置就在同目录
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# 把任何未捕获异常（含 Gradio 后台线程里的）写进 visual/error.log，便于排查
import threading as _threading      # noqa: E402
import traceback as _traceback      # noqa: E402

_ERROR_LOG = os.path.join(_HERE, "error.log")
_ORIG_EXCEPTHOOK = sys.excepthook
_ORIG_THREAD_EXCEPTHOOK = _threading.excepthook


def _log_exception(exc_type, exc, tb):
    """把未捕获异常写进 error.log。

    注意：必须同时转发给原始 excepthook 打印到控制台 —— 否则替换掉
    sys.excepthook 之后控制台会**什么错误都不显示**，
    排查时只能看到进程"莫名退出"。
    """
    try:
        with open(_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write("\n===== %s =====\n" % datetime.datetime.now().isoformat())
            f.write("".join(_traceback.format_exception(exc_type, exc, tb)))
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    _log_exception(exc_type, exc, tb)
    if _ORIG_EXCEPTHOOK is not None and _ORIG_EXCEPTHOOK is not _excepthook:
        _ORIG_EXCEPTHOOK(exc_type, exc, tb)      # 照常打印到控制台


def _thread_excepthook(args):
    _log_exception(args.exc_type, args.exc_value, args.exc_traceback)
    if _ORIG_THREAD_EXCEPTHOOK is not None and _ORIG_THREAD_EXCEPTHOOK is not _thread_excepthook:
        _ORIG_THREAD_EXCEPTHOOK(args)


sys.excepthook = _excepthook
_threading.excepthook = _thread_excepthook

# 仓库代码使用相对路径导入（configs / text 等），必须切到仓库根目录
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import numpy as np            # noqa: E402
import soundfile as sf        # noqa: E402
import torch                  # noqa: E402
from torch import LongTensor, no_grad   # noqa: E402

import commons                # noqa: E402
import utils                  # noqa: E402
from models import SynthesizerTrn       # noqa: E402
from text import text_to_sequence       # noqa: E402

try:
    from utils import tag_cjke
except Exception:             # noqa: BLE001
    tag_cjke = None


# ---------------------------------------------------------------------------
# 静音日志
#
# utils.py 第 15 行有 logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)，
# 于是 httpx / httpcore / PIL / matplotlib / asyncio 的每一条 DEBUG 日志都会打出来，
# 启动一次能刷几千行，真正的报错被彻底淹没。这里把日志级别调回 WARNING。
# ---------------------------------------------------------------------------
def _quiet_logging():
    import logging
    import warnings

    # 行缓冲：否则把输出重定向到文件时，启动信息会卡在缓冲区里看不到
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for h in root.handlers:
        h.setLevel(logging.WARNING)

    for name in ("httpx", "httpcore", "urllib3", "urllib3.connectionpool",
                 "PIL", "PIL.Image", "matplotlib", "numba", "jieba",
                 "asyncio", "gradio", "gradio_client", "uvicorn",
                 "uvicorn.error", "uvicorn.access", "multipart",
                 "filelock", "huggingface_hub", "numba.core"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 上游代码自身的弃用提示，与本服务无关
    warnings.filterwarnings("ignore", message=r".*weight_norm is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*TensorFloat-32.*")


_quiet_logging()

# ---------------------------------------------------------------------------
# 模型与配置：优先用 visual/ 下的，找不到再退回 OUTPUT_MODEL/
# ---------------------------------------------------------------------------
def _pick(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return cands[0]


MODEL_PATH = _pick(
    os.path.join(_HERE, "G_latest.pth"),
    os.path.join(_REPO, "OUTPUT_MODEL", "G_latest.pth"),
)
CONFIG_PATH = _pick(
    os.path.join(_HERE, "modified_finetune_speaker.json"),
    os.path.join(_REPO, "OUTPUT_MODEL", "config.json"),
)
OUTPUT_DIR = os.path.join(_HERE, "outputs")

# ---------------------------------------------------------------------------
# 阵营 -> 说话人名（说话人 ID 由配置文件里的 speakers 映射决定）
#   本模型的映射：ep_eva=0  ff_eva=1  su_eva=2  us_eva=3
# ---------------------------------------------------------------------------
FACTIONS = [
    ("盟军",     "us_eva"),
    ("苏联",     "su_eva"),
    ("焚风",     "ff_eva"),
    ("厄普西隆", "ep_eva"),
]
FACTION_TO_SPEAKER = dict(FACTIONS)

# ---------------------------------------------------------------------------
# 语言选项（仿官方 app 页面）
#   Mix 用 None 表示「不自动包裹整段」，交给下面的 build_tagged_text 处理
# ---------------------------------------------------------------------------
LANGUAGES = [
    ("English（英文）",     "[EN]"),
    ("简体中文",            "[ZH]"),
    ("日本語",              "[JA]"),
    ("Mix（多语言混合）",    None),
]
LANG_LABEL_TO_TAG = dict(LANGUAGES)

# 已存在的语言标记（用于判断用户是否手工标注过）
_TAG_RE = re.compile(r"\[(?:ZH|JA|EN|KR)\]")
# 用于自动识别兜底：是否含中文 / 日文假名 / 拉丁字母
_ZH_RE = re.compile(r"[\u4e00-\u9fa5]")
_JA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_EN_RE = re.compile(r"[a-zA-Z]")


# ---------------------------------------------------------------------------
def build_tagged_text(text, lang_label):
    """把用户输入变成带语言标记的文本。

    单一语言  -> 整段用该语言标记包裹
    Mix 模式  -> 已有标记则原样使用；否则自动逐字识别并补标记
    """
    text = (text or "").strip()
    if not text:
        return ""

    tag = LANG_LABEL_TO_TAG.get(lang_label)

    # ---- 单一语言：直接包裹整段 ----
    if tag is not None:
        return "%s%s%s" % (tag, text, tag)

    # ---- Mix 模式 ----
    if _TAG_RE.search(text):
        # 用户已按官方方式手工标注，原样尊重
        return text

    # 未标注 -> 自动识别
    if tag_cjke is not None:
        try:
            tagged = tag_cjke(text)
            if tagged and _TAG_RE.search(tagged):
                return tagged
        except Exception:
            pass  # 纯数字/纯符号等边角情况会让 tag_cjke 抛错，走下面的兜底

    # 兜底：按主要字符类型整段包裹
    if _JA_RE.search(text):
        return "[JA]" + text + "[JA]"
    if _ZH_RE.search(text):
        return "[ZH]" + text + "[ZH]"
    if _EN_RE.search(text):
        return "[EN]" + text + "[EN]"
    return "[ZH]" + text + "[ZH]"


# ---------------------------------------------------------------------------
class Synthesizer:
    """封装模型加载与推理。"""

    def __init__(self, model_path=MODEL_PATH, config_path=CONFIG_PATH, device=None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "找不到模型文件：%s\n"
                "请确认 visual/G_latest.pth 存在，或放在 OUTPUT_MODEL/ 下。" % model_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                "找不到配置文件：%s\n"
                "请确认 visual/modified_finetune_speaker.json 存在。" % config_path)

        self.model_path = model_path
        self.config_path = config_path

        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.hps = utils.get_hparams_from_file(config_path)
        self.net_g = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(self.device)
        self.net_g.eval()
        utils.load_checkpoint(model_path, self.net_g, None)

        self.sampling_rate = self.hps.data.sampling_rate
        self.speaker_ids = dict(self.hps.speakers)

    # -----------------------------------------------------------------
    def speaker_id(self, speaker_name):
        if speaker_name in self.speaker_ids:
            return self.speaker_ids[speaker_name]
        raise KeyError("模型里没有说话人 %r，可用：%s"
                       % (speaker_name, list(self.speaker_ids.keys())))

    # -----------------------------------------------------------------
    def synthesize(self, tagged_text, speaker_name, speed=1.0,
                   emotion=0.667, duration_randomness=0.6):
        """返回 (采样率, float32 波形数组)。"""
        sid = self.speaker_id(speaker_name)

        # text_to_sequence 内部会调用 cleaners 以及 print，这里屏蔽它的打印
        with contextlib.redirect_stdout(io.StringIO()):
            ids = text_to_sequence(tagged_text, self.hps.symbols,
                                   self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            ids = commons.intersperse(ids, 0)
        if not ids:
            raise ValueError("文本清洗后为空，无法合成。请确认输入内容与所选语言匹配。")

        stn = LongTensor(ids)
        # 倍速 -> VITS 的 length_scale：倍速越大，length_scale 越小（越快）
        length_scale = 1.0 / max(float(speed), 1e-6)

        with no_grad():
            x = stn.unsqueeze(0).to(self.device)
            x_len = LongTensor([stn.size(0)]).to(self.device)
            sid_t = LongTensor([sid]).to(self.device)
            audio = self.net_g.infer(
                x, x_len, sid=sid_t,
                noise_scale=float(emotion),
                noise_scale_w=float(duration_randomness),
                length_scale=length_scale,
            )[0][0, 0].data.cpu().float().numpy()
        del stn, x, x_len, sid_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.sampling_rate, audio


# ---------------------------------------------------------------------------
# 全局单例（Gradio 启动时加载一次）
# ---------------------------------------------------------------------------
SYNTH = None


def get_synth(device=None):
    global SYNTH
    if SYNTH is None:
        print("正在加载模型 ...")
        # 注意：必须显式传路径，不能依赖 __init__ 的默认参数
        #（默认参数在模块导入时就已经固定，之后改全局变量不会生效）
        SYNTH = Synthesizer(MODEL_PATH, CONFIG_PATH, device=device)
        print("  模型：%s" % SYNTH.model_path)
        print("  配置：%s" % SYNTH.config_path)
        print("  设备：%s   采样率：%d Hz" % (SYNTH.device, SYNTH.sampling_rate))
        print("  说话人：%s" % SYNTH.speaker_ids)
    return SYNTH


def save_wav(sr, audio, faction, text):
    """把生成的音频写成 16bit PCM wav，并追加一行历史记录。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe = re.sub(r"[^0-9A-Za-z_]+", "", FACTION_TO_SPEAKER.get(faction, faction)) or "out"
    name = "%s_%s.wav" % (safe, stamp)
    path = os.path.join(OUTPUT_DIR, name)
    pcm = np.clip(audio, -1.0, 1.0)
    sf.write(path, (pcm * 32767.0).astype(np.int16), sr, subtype="PCM_16")
    with open(os.path.join(OUTPUT_DIR, "history.tsv"), "a", encoding="utf-8") as f:
        f.write("%s\t%s\t%s\t%.2fs\t%s\n"
                % (stamp, faction,
                   text.replace("\t", " ").replace("\n", " ")[:200],
                   len(audio) / float(sr), name))
    return name


# ---------------------------------------------------------------------------
def tts_fn(text, faction, language, speed, emotion, dur_rnd, save_file):
    """Gradio 回调（外层兜底，避免任何异常直接把服务端打成 500）。"""
    try:
        return _tts_impl(text, faction, language, speed, emotion, dur_rnd, save_file)
    except Exception:  # noqa: BLE001
        import traceback
        return None, "❌ 内部错误：\n" + traceback.format_exc(limit=4), ""


def _tts_impl(text, faction, language, speed, emotion, dur_rnd, save_file):
    try:
        synth = get_synth()
    except Exception as e:  # noqa: BLE001
        return None, "❌ 模型未就绪：%s" % e, ""

    if not text or not text.strip():
        return None, "⚠️ 请输入要朗读的文字。", ""

    speaker = FACTION_TO_SPEAKER.get(faction)
    tagged = build_tagged_text(text, language)
    if not tagged:
        return None, "⚠️ 文本为空。", ""

    try:
        sr, audio = synth.synthesize(
            tagged, speaker,
            speed=float(speed),
            emotion=float(emotion),
            duration_randomness=float(dur_rnd),
        )
    except Exception as e:  # noqa: BLE001
        return None, "❌ 合成失败：%s" % e, tagged

    dur = len(audio) / float(sr)
    msg = "✅ %s（%s，id=%d） 语速 %.2fx  时长 %.2f 秒" % (
        faction, speaker, synth.speaker_id(speaker), float(speed), dur)

    if save_file:
        try:
            msg += "  已保存：visual/outputs/%s" % save_wav(sr, audio, faction, text)
        except Exception as e:  # noqa: BLE001
            msg += "  （保存失败：%s）" % e

    pcm = np.clip(audio, -1.0, 1.0)
    return (sr, (pcm * 32767.0).astype(np.int16)), msg, tagged


# ===========================================================================
# 前端外观（纯展示层）
#
# 说明：这里只改「长什么样」，不改「能干什么」。
#   * 组件种类、数量、顺序、label、默认值、事件绑定、api_name 全部保持不变
#   * 因此 visual/app.py 的 /tts 接口签名、启动参数、日志输出都不受影响
#   * 主题走 gr.themes.Base().set()，深/浅色两套变量都指向同一套深色值，
#     这样用户点右上角的明暗切换按钮也不会破坏外观
#
# 设计取向：红警军用指挥终端 + 心灵终结四阵营链路。
# 金属切角、战术网格、扫描雷达与声纹动画全部使用本地 CSS/SVG 实现；
# 阵营切换、字数统计与传输反馈只运行前端 JS，不改变推理参数和 API。
# ===========================================================================

TECH_CSS = """
/* =====================================================================
   HUD · 深空指挥台主题
   ===================================================================== */

:root {
  --hud-panel:     rgba(12, 23, 37, .92);
  --hud-panel-2:   rgba(7, 14, 24, .88);
  --hud-line:      rgba(56, 189, 248, .16);
  --hud-line-soft: rgba(56, 189, 248, .10);
  --hud-cyan:      #22d3ee;
  --hud-sky:       #38bdf8;
  --hud-text:      #d7e6f5;
  --hud-muted:     #6f8aa3;
  --hud-mono:      "JetBrains Mono", "Consolas", "SFMono-Regular", monospace;
}

/* ---------------- 画布 ---------------- */
html, body, gradio-app { background-color: #050a12 !important; }
.gradio-container {
  background: transparent !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 22px 22px 30px !important;
  position: relative;
  z-index: 1;
  font-size: 14.5px !important;
}
body, .gradio-container, .gradio-container .prose * { color: var(--hud-text); }
body {
  background-color: #050a12 !important;
  background-image:
    radial-gradient(1100px 560px at 8% -12%, rgba(34, 211, 238, .17), transparent 62%),
    radial-gradient(980px 520px at 92% -6%, rgba(167, 139, 250, .14), transparent 58%),
    radial-gradient(820px 420px at 50% 112%, rgba(14, 165, 233, .10), transparent 60%),
    linear-gradient(180deg, #050a12 0%, #060d18 42%, #04070e 100%) !important;
  background-attachment: fixed !important;
}
/* 网格底纹 */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(56, 189, 248, .05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, .05) 1px, transparent 1px);
  background-size: 46px 46px;
  -webkit-mask-image: radial-gradient(circle at 50% 0%, #000 0%, rgba(0, 0, 0, .28) 55%, transparent 82%);
          mask-image: radial-gradient(circle at 50% 0%, #000 0%, rgba(0, 0, 0, .28) 55%, transparent 82%);
}

/* ---------------- 顶部 HERO ---------------- */
.hud-hero {
  position: relative;
  display: flex;
  align-items: center;
  gap: 26px;
  padding: 24px 28px;
  margin-bottom: 16px;
  border-radius: 18px;
  border: 1px solid rgba(56, 189, 248, .22);
  background:
    linear-gradient(135deg, rgba(12, 27, 44, .95) 0%, rgba(7, 15, 26, .92) 55%, rgba(13, 21, 38, .95) 100%);
  box-shadow:
    0 26px 64px -36px rgba(0, 0, 0, .95),
    inset 0 1px 0 rgba(125, 211, 252, .10);
  overflow: hidden;
}
/* 顶部扫光线 */
.hud-hero::before {
  content: "";
  position: absolute;
  left: 0; right: 0; top: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(34, 211, 238, .9), transparent);
  animation: hud-scan 5.5s ease-in-out infinite;
}
@keyframes hud-scan {
  0%, 100% { opacity: .25; transform: translateX(-28%); }
  50%      { opacity: 1;   transform: translateX(28%); }
}
.hud-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font: 600 10.5px/1 var(--hud-mono);
  letter-spacing: .22em;
  color: #7dd3fc;
  border: 1px solid rgba(56, 189, 248, .35);
  border-radius: 999px;
  padding: 7px 13px;
  background: rgba(14, 116, 144, .16);
}
.hud-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--hud-cyan);
  box-shadow: 0 0 10px var(--hud-cyan);
  animation: hud-pulse 1.9s ease-in-out infinite;
}
@keyframes hud-pulse {
  0%, 100% { opacity: .35; transform: scale(.8); }
  50%      { opacity: 1;   transform: scale(1.2); }
}
.hud-hero h1 {
  margin: 12px 0 8px;
  text-shadow: 0 0 26px rgba(34, 211, 238, .22);
  font-size: 29px;
  font-weight: 800;
  line-height: 1.28;
  letter-spacing: .02em;
  color: #eaf6ff;
}
.hud-hero h1 em {
  font-style: normal;
  background: linear-gradient(92deg, #22d3ee, #60a5fa 55%, #a78bfa);
  -webkit-background-clip: text;
          background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hud-hero .hud-sep { color: #2f5573; margin: 0 6px; font-weight: 400; }
.hud-chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.hud-chip {
  font: 600 12.5px/1 var(--hud-mono);
  letter-spacing: .08em;
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, .25);
  background: rgba(148, 163, 184, .06);
  color: #9fb6cc;
  white-space: nowrap;
}
.hud-chip.fc-us { color: #7dd3fc; border-color: rgba(56, 189, 248, .45); background: rgba(56, 189, 248, .10); }
.hud-chip.fc-su { color: #fca5a5; border-color: rgba(248, 113, 113, .45); background: rgba(248, 113, 113, .10); }
.hud-chip.fc-ff { color: #fde68a; border-color: rgba(250, 204, 21, .40); background: rgba(250, 204, 21, .10); }
.hud-chip.fc-ep { color: #c4b5fd; border-color: rgba(167, 139, 250, .45); background: rgba(167, 139, 250, .10); }

/* 雷达 */
.hud-radar { width: 150px; height: 150px; flex: 0 0 150px; margin-left: auto; opacity: .95; }
.hud-radar .sweep { transform-origin: 74px 74px; animation: hud-rotate 6.5s linear infinite; }
@keyframes hud-rotate { to { transform: rotate(360deg); } }
.hud-radar .blip { animation: hud-blip 2.6s ease-in-out infinite; }
.hud-radar .blip.b2 { animation-delay: .9s; }
.hud-radar .blip.b3 { animation-delay: 1.7s; }
@keyframes hud-blip {
  0%, 100% { opacity: .15; r: 2.2; }
  45%      { opacity: 1;   r: 3.2; }
}

/* ---------------- 面板 ---------------- */
.hud-main { gap: 16px !important; align-items: stretch !important; }
.hud-card {
  position: relative;
  padding: 18px 20px 20px !important;
  border-radius: 16px !important;
  border: 1px solid rgba(56, 189, 248, .22) !important;
  background: linear-gradient(165deg, var(--hud-panel), var(--hud-panel-2)) !important;
  box-shadow:
    0 22px 52px -30px rgba(0, 0, 0, .95),
    inset 0 1px 0 rgba(125, 211, 252, .06) !important;
  backdrop-filter: blur(12px);
}
/* HUD 角标 */
.hud-card::before, .hud-card::after {
  content: "";
  position: absolute;
  width: 14px; height: 14px;
  border: 0 solid rgba(34, 211, 238, .55);
  pointer-events: none;
}
.hud-card::before {
  top: -1px; left: -1px;
  border-top-width: 2px; border-left-width: 2px;
  border-top-left-radius: 12px;
}
.hud-card::after {
  bottom: -1px; right: -1px;
  border-bottom-width: 2px; border-right-width: 2px;
  border-bottom-right-radius: 12px;
}

/* 分区标题 */
.hud-sec { display: flex; align-items: center; gap: 10px; margin: 0 0 14px; }
.hud-sec .n {
  font: 800 12.5px/1 var(--hud-mono);
  color: #04121b;
  background: linear-gradient(135deg, #22d3ee, #38bdf8);
  padding: 7px 9px;
  border-radius: 6px;
  letter-spacing: .04em;
}
.hud-sec .t { font-size: 15.5px; font-weight: 700; letter-spacing: .14em; color: #dcefff; }
.hud-sec .t small {
  display: block;
  margin-top: 4px;
  font: 500 11px/1 var(--hud-mono);
  letter-spacing: .2em;
  color: #5f7b96;
  text-transform: uppercase;
}
.hud-sec::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(56, 189, 248, .34), transparent);
}

/* ---------------- 表单控件 ---------------- */
/* 设计原则：每个控件只画一层盒子。
   Gradio 把输入框的边框画在 .form（文本框/滑块）和 .wrap（下拉框）上，
   所以把盒子统一放在这两个类上，.block 只当透明容器，避免"框套框"。 */
.hud-card .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.hud-card .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* 下拉框的盒子画在 .wrap 上（它才是下拉的可点区域） */
.hud-card .wrap {
  background: #050e18 !important;
  border: 1px solid rgba(56, 189, 248, .30) !important;
  border-radius: 9px !important;
  box-shadow: none !important;
  transition: border-color .16s ease, box-shadow .16s ease;
}
.hud-card .wrap:hover { border-color: rgba(56, 189, 248, .46) !important; }
.hud-card .wrap:focus-within {
  border-color: rgba(34, 211, 238, .85) !important;
  box-shadow:
    0 0 0 3px rgba(34, 211, 238, .12),
    0 0 20px -6px rgba(34, 211, 238, .55) !important;
}
.hud-card .wrap-inner,
.hud-card .form > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.hud-card .info, .hud-card span.info, .hud-card div.info,
.hud-card label > div {
  color: #6d8aa4 !important;
  font-size: 12px !important;
  line-height: 1.6 !important;
  letter-spacing: .02em !important;
}
/* 文本框自带一圈盒子，label 留在框外，和下拉框保持同一种观感 */
.hud-card textarea,
.hud-card input[type="text"],
.hud-card input[type="number"] {
  background: #050e18 !important;
  border: 1px solid rgba(56, 189, 248, .30) !important;
  border-radius: 9px !important;
  box-shadow: none !important;
  color: #e6f4ff !important;
  font-size: 15px !important;
  line-height: 1.75 !important;
  padding: 9px 12px !important;
  transition: border-color .16s ease, box-shadow .16s ease;
}
.hud-card textarea:hover,
.hud-card input[type="text"]:hover { border-color: rgba(56, 189, 248, .46) !important; }
.hud-card textarea:focus,
.hud-card input[type="text"]:focus,
.hud-card input[type="number"]:focus {
  outline: none !important;
  border-color: rgba(34, 211, 238, .85) !important;
  box-shadow:
    0 0 0 3px rgba(34, 211, 238, .12),
    0 0 20px -6px rgba(34, 211, 238, .55) !important;
}
.hud-card input[type="range"], .hud-card input[type="checkbox"] { accent-color: var(--hud-cyan); }
.hud-card .label-wrap, .hud-card button.label-wrap {
  color: #9dc0dc !important;
  font-size: 14px !important;
  letter-spacing: .06em !important;
}
/* 所有组件的主标签（要朗读的文字 / 语音倍速 / 状态 / 生成结果 …） */
.hud-card [data-testid="block-info"], .hud-card .block-title,
.hud-card [data-testid="block-label"], .hud-card [data-testid="block-label"] span {
  font-size: 14px !important;
  font-weight: 600 !important;
  color: #cfe6f7 !important;
  letter-spacing: .02em !important;
}
/* 滑块那一行：标签与边框留出呼吸距离 */
.hud-card .wrap > .head {
  padding: 4px 14px 2px !important;
  gap: 10px !important;
}
.hud-card .wrap > .head label { align-items: center !important; }
.hud-card input[type="range"] {
  margin: 0 14px 8px !important;
  width: calc(100% - 28px) !important;
}

/* 下拉框：左列没有说明文字时，把选择框推到底部，与右列的选择框齐平 */
.hud-card .block:has(.wrap-inner) {
  display: flex !important;
  flex-direction: column !important;
}
.hud-card .block:has(.wrap-inner) > .container {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
}
.hud-card .block:has(.wrap-inner) .wrap { margin-top: auto !important; }

/* 下拉菜单浮层 */
.hud-card ul.options, .hud-card .options {
  background: #0a1523 !important;
  border: 1px solid rgba(56, 189, 248, .22) !important;
  border-radius: 10px !important;
  box-shadow: 0 20px 44px -20px rgba(0, 0, 0, .95) !important;
}
.hud-card ul.options li, .hud-card .options .item {
  color: #cfe6f7 !important;
  font-size: 13px !important;
}
.hud-card ul.options li:hover, .hud-card .options .item:hover,
.hud-card ul.options li.selected, .hud-card .options .item.selected {
  background: rgba(14, 116, 144, .35) !important;
  color: #eaf6ff !important;
}

/* ---------------- 主按钮 ---------------- */
button.hud-btn {
  min-height: 46px !important;
  border: none !important;
  border-radius: 10px !important;
  background: linear-gradient(96deg, #0891b2 0%, #22d3ee 48%, #38bdf8 100%) !important;
  color: #04121b !important;
  font-weight: 800 !important;
  font-size: 14px !important;
  letter-spacing: .16em !important;
  box-shadow:
    0 0 0 1px rgba(34, 211, 238, .5),
    0 16px 36px -18px rgba(34, 211, 238, .95) !important;
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
button.hud-btn:hover {
  transform: translateY(-1px);
  filter: brightness(1.08);
  box-shadow:
    0 0 0 1px rgba(34, 211, 238, .8),
    0 20px 44px -18px rgba(34, 211, 238, 1) !important;
}
button.hud-btn:active { transform: translateY(0); }

/* ---------------- 输出区 ---------------- */
.hud-card audio, .hud-card .audio-player, .hud-card .waveform-container {
  background: rgba(3, 9, 16, .8) !important;
  border-radius: 10px !important;
}
.hud-readout textarea {
  color: #7dd3fc !important;
  font-family: var(--hud-mono) !important;
  font-size: 13.5px !important;
  letter-spacing: .02em !important;
}

/* ---------------- 示例台词 ---------------- */
#hud-examples { margin-top: 16px !important; }
#hud-examples > div { border: none !important; background: transparent !important; }
#hud-examples table {
  background: transparent !important;
  border: none !important;
  border-collapse: separate !important;
  border-spacing: 0 6px !important;
  width: 100% !important;
}
#hud-examples thead { display: none; }
#hud-examples tbody tr td {
  background: rgba(16, 31, 48, .92) !important;
  border: 1px solid rgba(56, 189, 248, .14) !important;
  border-left: none !important;
  color: #9fb6cc !important;
  font-size: 12.5px !important;
  padding: 9px 12px !important;
  text-align: left !important;
  transition: background .15s ease, color .15s ease, border-color .15s ease;
}
#hud-examples tbody tr td:first-child {
  border-left: 1px solid rgba(56, 189, 248, .14) !important;
  border-radius: 8px 0 0 8px !important;
  font-family: var(--hud-mono) !important;
  font-size: 12px !important;
  color: #c3d8ea !important;
}
#hud-examples tbody tr td:last-child {
  border-radius: 0 8px 8px 0 !important;
}
/* 后四列只是参数回显，弱化处理 */
#hud-examples tbody tr td:nth-child(n+4) {
  color: #4d6a80 !important;
  font-family: var(--hud-mono) !important;
  font-size: 11px !important;
  text-align: center !important;
}
#hud-examples tbody tr:hover td {
  background: rgba(14, 116, 144, .26) !important;
  border-color: rgba(34, 211, 238, .38) !important;
  color: #eaf6ff !important;
  cursor: pointer;
}
#hud-examples tbody tr:hover td:nth-child(n+4) { color: #7dd3fc !important; }

/* ---------------- 右侧提示条 ---------------- */
#hud-hint { margin-top: auto !important; border: none !important; background: transparent !important; }
.hud-card textarea { resize: none !important; }
.hud-hint {
  margin-top: auto;
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px dashed rgba(56, 189, 248, .22);
  background: rgba(8, 18, 30, .55);
  font-size: 12.5px;
  line-height: 1.8;
  color: #7b96ae;
}
.hud-hint .k {
  display: inline-block;
  margin-right: 8px;
  font: 800 9.5px/1 var(--hud-mono);
  letter-spacing: .18em;
  color: #7dd3fc;
}
.hud-hint code {
  background: rgba(56, 189, 248, .10);
  border: 1px solid rgba(56, 189, 248, .16);
  border-radius: 4px;
  padding: 0 4px;
  color: #7dd3fc;
  font-size: 10.5px;
}

/* ---------------- 页脚 ---------------- */
.hud-foot {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
  margin-top: 18px;
  padding-top: 12px;
  border-top: 1px solid var(--hud-line-soft);
  font: 11px/1.7 var(--hud-mono);
  letter-spacing: .06em;
  color: #4d6580;
}
.hud-foot code {
  background: rgba(56, 189, 248, .10);
  border: 1px solid rgba(56, 189, 248, .18);
  border-radius: 4px;
  padding: 1px 5px;
  color: #7dd3fc;
  font-size: 10.5px;
}

/* Gradio 自带页脚压暗，不隐藏（保留 "Use via API" 入口） */
.gradio-container footer { background: transparent !important; color: #3f556e !important; }
.gradio-container footer a { color: #526b85 !important; }

/* ---------------- 响应式 ---------------- */
@media (max-width: 980px) {
  /* 窄屏下两张面板改为上下堆叠，避免被挤成细条 */
  .hud-main { flex-direction: column !important; }
  .hud-main > .column, .hud-main > div {
    flex: 1 1 100% !important;
    min-width: 100% !important;
    width: 100% !important;
  }
  #hud-examples tbody tr td { padding: 8px 9px !important; font-size: 12px !important; }
}
@media (max-width: 860px) {
  .hud-hero { flex-direction: column; align-items: flex-start; gap: 14px; padding: 20px 20px; }
  .hud-radar { display: none; }
  .hud-hero h1 { font-size: 23px; }
  .hud-card { padding: 15px 15px 17px !important; }
}
"""

# ---------------------------------------------------------------------------
# MENTAL OMEGA / BATTLE CONTROL：在基础重置之上应用正式界面主题
# ---------------------------------------------------------------------------
TECH_CSS += r"""
:root {
  --mo-accent: #65bfff;
  --mo-accent-rgb: 101, 191, 255;
  --mo-accent-2: #b9e3ff;
  --mo-ink: #e8edf0;
  --mo-muted: #87949a;
  --mo-dim: #566168;
  --mo-panel: #11171a;
  --mo-panel-deep: #090d0f;
  --mo-line: rgba(166, 183, 190, .18);
  --mo-font: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  --mo-mono: "Cascadia Mono", "Consolas", "SFMono-Regular", monospace;
}
body[data-faction="su"] { --mo-accent: #ff6257; --mo-accent-rgb: 255, 98, 87; --mo-accent-2: #ffc0a8; }
body[data-faction="ff"] { --mo-accent: #51e0bd; --mo-accent-rgb: 81, 224, 189; --mo-accent-2: #c0ffec; }
body[data-faction="ep"] { --mo-accent: #c88aff; --mo-accent-rgb: 200, 138, 255; --mo-accent-2: #ead0ff; }

html, body, gradio-app { background: #070a0b !important; }
body {
  color: var(--mo-ink) !important;
  background:
    radial-gradient(1000px 500px at 14% -10%, rgba(var(--mo-accent-rgb), .10), transparent 65%),
    linear-gradient(145deg, #101619 0%, #070a0b 42%, #0b0f11 100%) !important;
  transition: background-color .45s ease;
}
body::before {
  background-image:
    linear-gradient(rgba(var(--mo-accent-rgb), .035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--mo-accent-rgb), .035) 1px, transparent 1px),
    linear-gradient(rgba(var(--mo-accent-rgb), .018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--mo-accent-rgb), .018) 1px, transparent 1px) !important;
  background-size: 80px 80px, 80px 80px, 16px 16px, 16px 16px !important;
  -webkit-mask-image: linear-gradient(to bottom, #000, rgba(0,0,0,.35) 75%, transparent) !important;
          mask-image: linear-gradient(to bottom, #000, rgba(0,0,0,.35) 75%, transparent) !important;
}
body::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: .20;
  background: repeating-linear-gradient(0deg, transparent 0 3px, rgba(255,255,255,.018) 3px 4px);
}
.gradio-container {
  max-width: 1340px !important;
  padding: 20px 26px 30px !important;
  color: var(--mo-ink) !important;
  font-family: var(--mo-font) !important;
}
.gradio-container .prose, .gradio-container .prose * { color: inherit; }
.gradio-container footer { opacity: .55; }

/* 顶部：作战链路标识 + 雷达 + 状态灯 */
.mo-command {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 210px;
  min-height: 228px;
  margin-bottom: 12px;
  border: 1px solid rgba(var(--mo-accent-rgb), .34);
  background:
    linear-gradient(100deg, rgba(var(--mo-accent-rgb), .08), transparent 46%),
    repeating-linear-gradient(135deg, rgba(255,255,255,.013) 0 2px, transparent 2px 9px),
    #101619;
  clip-path: polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 22px 100%, 0 calc(100% - 22px));
  box-shadow: inset 0 0 70px rgba(0,0,0,.40), 0 20px 50px rgba(0,0,0,.28);
  animation: mo-boot .65s cubic-bezier(.2,.8,.2,1) both;
}
.mo-command::before {
  content: "";
  position: absolute;
  inset: 0 auto auto 0;
  width: 42%;
  height: 3px;
  background: linear-gradient(90deg, var(--mo-accent), transparent);
  box-shadow: 0 0 20px rgba(var(--mo-accent-rgb), .55);
}
.mo-command-main { position: relative; z-index: 2; display: flex; flex-direction: column; justify-content: center; padding: 27px 30px 24px; }
.mo-eyebrow {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  color: var(--mo-accent);
  font: 700 10px/1.3 var(--mo-mono);
  letter-spacing: .19em;
}
.mo-online {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 9px;
  border: 1px solid rgba(var(--mo-accent-rgb), .35);
  background: rgba(var(--mo-accent-rgb), .07);
}
.mo-online::before {
  content: "";
  width: 6px;
  height: 6px;
  background: var(--mo-accent);
  box-shadow: 0 0 10px var(--mo-accent);
  animation: mo-signal 1.8s steps(2, end) infinite;
}
.mo-title {
  margin: 18px 0 7px !important;
  color: #f3f5f4 !important;
  font: 800 clamp(28px, 4vw, 46px)/.98 var(--mo-font) !important;
  letter-spacing: -.035em !important;
  text-transform: uppercase;
  text-shadow: 3px 3px 0 rgba(var(--mo-accent-rgb), .12);
}
.mo-title span { color: var(--mo-accent); transition: color .35s ease; }
.mo-subtitle {
  margin: 0;
  color: #aab4b8;
  font-size: 14px;
  letter-spacing: .16em;
}
.mo-subtitle b { color: var(--mo-ink); font-weight: 650; }
.mo-radar-wrap {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 9px;
  padding: 14px 0 16px;
  border-left: 1px solid var(--mo-line);
  background: rgba(0,0,0,.15);
}
.mo-radar { flex: 0 0 164px; width: 164px; height: 164px; overflow: visible; }
.mo-radar .mo-sweep { transform-origin: 80px 80px; animation: mo-rotate 7s linear infinite; }
.mo-radar .mo-blip { animation: mo-blip 2.4s ease-in-out infinite; }
.mo-radar .mo-blip:nth-of-type(2) { animation-delay: -.8s; }
.mo-radar .mo-blip:nth-of-type(3) { animation-delay: -1.6s; }
.mo-radar-label {
  position: static;
  flex: 0 0 auto;
  color: var(--mo-accent);
  font: 11px/1 var(--mo-mono);
  letter-spacing: .16em;
  white-space: nowrap;
}

/* 指挥状态条 */
.mo-statusbar {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  margin-bottom: 16px;
  border: 1px solid var(--mo-line);
  background: #0b1012;
  animation: mo-boot .65s .08s cubic-bezier(.2,.8,.2,1) both;
}
.mo-stat { position: relative; padding: 11px 16px 10px 31px; border-right: 1px solid var(--mo-line); }
.mo-stat:last-child { border-right: 0; }
.mo-stat::before {
  content: "";
  position: absolute;
  left: 15px;
  top: 17px;
  width: 5px;
  height: 5px;
  background: var(--mo-accent);
  opacity: .8;
}
.mo-stat small { display: block; color: var(--mo-dim); font: 8px/1.3 var(--mo-mono); letter-spacing: .16em; }
.mo-stat strong { display: block; margin-top: 3px; color: #bdc7ca; font: 600 11px/1.3 var(--mo-mono); letter-spacing: .05em; }

/* 主控制台 */
.mo-main { gap: 16px !important; align-items: stretch !important; }
.mo-card {
  position: relative;
  min-width: 0;
  padding: 20px 22px 22px !important;
  border: 1px solid var(--mo-line) !important;
  border-top-color: rgba(var(--mo-accent-rgb), .38) !important;
  border-radius: 2px !important;
  background:
    linear-gradient(180deg, rgba(var(--mo-accent-rgb), .035), transparent 110px),
    var(--mo-panel) !important;
  box-shadow: inset 0 1px rgba(255,255,255,.025), 0 14px 38px rgba(0,0,0,.22) !important;
  animation: mo-boot .6s .15s cubic-bezier(.2,.8,.2,1) both;
}
.mo-card::before, .mo-card::after {
  content: "";
  position: absolute;
  z-index: 2;
  width: 20px;
  height: 20px;
  pointer-events: none;
}
.mo-card::before { left: -1px; top: -1px; border-left: 2px solid var(--mo-accent); border-top: 2px solid var(--mo-accent); }
.mo-card::after { right: -1px; bottom: -1px; border-right: 2px solid var(--mo-accent); border-bottom: 2px solid var(--mo-accent); }
.mo-section { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.mo-section-index {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  color: #071013;
  background: var(--mo-accent);
  font: 800 13px/1 var(--mo-mono);
  clip-path: polygon(0 0, 100% 0, 100% 72%, 72% 100%, 0 100%);
  transition: background .35s ease;
}
.mo-section-text { min-width: 0; }
.mo-section-text b { display: block; color: #e0e6e8; font-size: 14px; letter-spacing: .13em; }
.mo-section-text small { display: block; margin-top: 3px; color: var(--mo-dim); font: 9px/1 var(--mo-mono); letter-spacing: .18em; }
.mo-section-line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--mo-line), transparent); }
.mo-section-code { color: var(--mo-accent); font: 9px/1 var(--mo-mono); letter-spacing: .1em; }
.mo-input-meta { display: flex; justify-content: flex-end; margin: -7px 2px 14px; color: var(--mo-dim); font: 9px/1.3 var(--mo-mono); letter-spacing: .08em; }
#mo-char-count { color: var(--mo-accent); }

/* Gradio 控件：保持单层描边，避免框套框 */
.mo-card .block, .mo-card .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.mo-card .wrap {
  background: #090e10 !important;
  border: 1px solid #303b3f !important;
  border-radius: 2px !important;
  box-shadow: inset 3px 0 0 rgba(var(--mo-accent-rgb), .15) !important;
  transition: border-color .2s ease, box-shadow .2s ease, background .2s ease;
}
.mo-card .wrap:hover { border-color: rgba(var(--mo-accent-rgb), .5) !important; }
.mo-card .wrap:focus-within {
  border-color: var(--mo-accent) !important;
  box-shadow: inset 3px 0 0 var(--mo-accent), 0 0 0 2px rgba(var(--mo-accent-rgb), .10) !important;
}
.mo-card .wrap-inner, .mo-card .form > div { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.mo-card textarea, .mo-card input[type="text"], .mo-card input[type="number"] {
  background: #090e10 !important;
  border: 1px solid #303b3f !important;
  border-radius: 2px !important;
  color: #e3e9eb !important;
  box-shadow: inset 3px 0 0 rgba(var(--mo-accent-rgb), .18) !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
  transition: border-color .2s ease, box-shadow .2s ease, background .2s ease;
}
#mo-input textarea { min-height: 132px !important; padding: 13px 15px !important; font-size: 15px !important; }
.mo-card textarea:hover, .mo-card input[type="text"]:hover { border-color: rgba(var(--mo-accent-rgb), .5) !important; }
.mo-card textarea:focus, .mo-card input[type="text"]:focus, .mo-card input[type="number"]:focus {
  outline: none !important;
  border-color: var(--mo-accent) !important;
  background: #0b1215 !important;
  box-shadow: inset 3px 0 0 var(--mo-accent), 0 0 0 2px rgba(var(--mo-accent-rgb), .10) !important;
}
.mo-card textarea::placeholder, .mo-card input::placeholder { color: #4c585d !important; }
.mo-card .label-wrap, .mo-card button.label-wrap,
.mo-card [data-testid="block-info"], .mo-card .block-title,
.mo-card [data-testid="block-label"], .mo-card [data-testid="block-label"] span {
  color: #b7c2c6 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: .045em !important;
}
.mo-card .info, .mo-card span.info, .mo-card div.info, .mo-card label > div {
  color: #647278 !important;
  font-size: 11px !important;
  line-height: 1.55 !important;
}
.mo-card .wrap > .head { padding: 4px 13px 2px !important; }
.mo-card input[type="range"] { width: calc(100% - 26px) !important; margin: 4px 13px 10px !important; accent-color: var(--mo-accent); }
.mo-card input[type="checkbox"] { accent-color: var(--mo-accent); }
.mo-card ul.options, .mo-card .options {
  background: #11191c !important;
  border: 1px solid rgba(var(--mo-accent-rgb), .35) !important;
  border-radius: 2px !important;
  box-shadow: 0 16px 35px rgba(0,0,0,.65) !important;
}
.mo-card ul.options li, .mo-card .options .item { color: #bdc8cb !important; }
.mo-card ul.options li:hover, .mo-card .options .item:hover,
.mo-card ul.options li.selected, .mo-card .options .item.selected {
  color: #fff !important;
  background: rgba(var(--mo-accent-rgb), .16) !important;
}
.mo-card details, .mo-card .accordion {
  border: 1px solid #293236 !important;
  border-radius: 2px !important;
  background: #0c1113 !important;
}

/* 主按钮：机械按压 + 扫描高光 */
button.mo-generate {
  position: relative;
  overflow: hidden;
  min-height: 54px !important;
  margin-top: 15px !important;
  border: 1px solid var(--mo-accent) !important;
  border-radius: 1px !important;
  background:
    linear-gradient(110deg, rgba(0,0,0,.18), transparent 45%),
    var(--mo-accent) !important;
  color: #071012 !important;
  font: 800 13px/1 var(--mo-mono) !important;
  letter-spacing: .14em !important;
  clip-path: polygon(0 0, calc(100% - 13px) 0, 100% 13px, 100% 100%, 13px 100%, 0 calc(100% - 13px));
  box-shadow: 0 8px 24px rgba(var(--mo-accent-rgb), .14) !important;
  transition: transform .14s ease, filter .2s ease, box-shadow .2s ease;
}
button.mo-generate::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(105deg, transparent 30%, rgba(255,255,255,.45) 48%, transparent 65%);
  transform: translateX(-120%);
  transition: transform .55s ease;
}
button.mo-generate:hover { transform: translateY(-2px); filter: brightness(1.08); box-shadow: 0 12px 30px rgba(var(--mo-accent-rgb), .25) !important; }
button.mo-generate:hover::after { transform: translateX(120%); }
button.mo-generate:active { transform: translateY(1px); }
body.is-busy button.mo-generate { animation: mo-transmit .8s ease-in-out infinite alternate; pointer-events: none; }

/* 声纹监视器 */
.mo-monitor {
  position: relative;
  height: 105px;
  margin-bottom: 17px;
  overflow: hidden;
  border: 1px solid #2a3539;
  background:
    linear-gradient(rgba(var(--mo-accent-rgb), .06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--mo-accent-rgb), .06) 1px, transparent 1px),
    #080d0f;
  background-size: 20px 20px;
}
.mo-monitor-head { position: absolute; z-index: 2; left: 11px; right: 11px; top: 9px; display: flex; justify-content: space-between; color: var(--mo-dim); font: 8px/1 var(--mo-mono); letter-spacing: .13em; }
#mo-link-state { color: var(--mo-accent); }
.mo-wave { position: absolute; inset: 31px 14px 12px; display: flex; align-items: center; justify-content: center; gap: 5px; }
.mo-wave i {
  display: block;
  width: 3px;
  height: 44px;
  background: var(--mo-accent);
  opacity: .40;
  transform: scaleY(.12);
  box-shadow: 0 0 7px rgba(var(--mo-accent-rgb), .28);
  animation: mo-wave 2.8s calc(var(--i) * -.11s) ease-in-out infinite;
}
body.is-busy .mo-wave i { opacity: .9; animation-duration: .72s; }
body.is-complete .mo-monitor { animation: mo-complete .75s ease-out; }
.mo-card audio, .mo-card .audio-player, .mo-card .waveform-container { background: #090e10 !important; border-radius: 2px !important; }
.mo-card .mo-readout textarea { color: var(--mo-accent-2) !important; font-family: var(--mo-mono) !important; font-size: 12px !important; }
#mo-tagged textarea { color: #8fa0a6 !important; font-size: 11px !important; }
.mo-hint { margin-top: 14px; padding: 10px 12px; border-left: 2px solid rgba(var(--mo-accent-rgb), .35); background: rgba(0,0,0,.13); color: #647278; font-size: 11px; line-height: 1.7; }
.mo-hint b { color: #9eacb1; font-weight: 600; }
.mo-hint code { color: var(--mo-accent); font-family: var(--mo-mono); }

/* 示例即战术通讯记录，点击行为仍由 gr.Examples 原生处理 */
#mo-examples {
  margin-top: 16px !important;
  border: 1px solid rgba(var(--mo-accent-rgb), .22) !important;
  border-radius: 2px !important;
  background: #0d1214 !important;
  overflow: hidden !important;
  animation: mo-boot .6s .22s cubic-bezier(.2,.8,.2,1) both;
}
#mo-examples .label {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 14px 17px !important;
  border-bottom: 1px solid rgba(var(--mo-accent-rgb), .20) !important;
  background: linear-gradient(90deg, rgba(var(--mo-accent-rgb), .09), transparent 62%) !important;
  color: #d9e2e5 !important;
  font: 700 15px/1.5 var(--mo-font) !important;
  letter-spacing: .06em !important;
}
#mo-examples .label svg { width: 18px; height: 18px; flex: 0 0 18px; color: var(--mo-accent); }
#mo-examples .table-wrap { width: 100%; overflow-x: auto; border-top: 0 !important; }
#mo-examples table { width: 100% !important; border-collapse: collapse !important; background: transparent !important; }
#mo-examples thead { background: #151e21 !important; }
#mo-examples thead th {
  padding: 13px 16px !important;
  border-right: 1px solid #253034 !important;
  border-bottom: 1px solid rgba(var(--mo-accent-rgb), .22) !important;
  color: #a9b6bb !important;
  font: 600 13px/1.45 var(--mo-font) !important;
  letter-spacing: .045em !important;
  text-transform: none !important;
  white-space: nowrap;
}
#mo-examples thead th:last-child { border-right: 0 !important; }
#mo-examples tbody tr td {
  padding: 10px 12px !important;
  border: 0 !important;
  border-bottom: 1px solid #20292c !important;
  background: transparent !important;
  color: #98a7ac !important;
  font-size: 11px !important;
  transition: color .18s ease, background .18s ease, padding-left .18s ease;
}
#mo-examples tbody tr td:first-child { border-left: 2px solid transparent !important; font-family: var(--mo-mono); }
#mo-examples tbody tr td:nth-child(n+4) { color: #536166 !important; font-family: var(--mo-mono); text-align: center; }
#mo-examples tbody tr:hover td { color: #e5edef !important; background: rgba(var(--mo-accent-rgb), .07) !important; cursor: pointer; }
#mo-examples tbody tr:hover td:first-child { border-left-color: var(--mo-accent) !important; padding-left: 17px !important; }
#mo-examples tbody tr:hover td:nth-child(n+4) { color: var(--mo-accent) !important; }

.mo-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-top: 17px; padding: 12px 2px 0; border-top: 1px solid var(--mo-line); color: #4e5c61; font: 9px/1.6 var(--mo-mono); letter-spacing: .08em; }
.mo-footer b { color: var(--mo-accent); font-weight: 500; }
.mo-footer code { color: #819096; font-family: var(--mo-mono); }

/* 页面进入 / 交互反馈 */
@keyframes mo-boot { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes mo-rotate { to { transform: rotate(360deg); } }
@keyframes mo-signal { 50% { opacity: .25; } }
@keyframes mo-blip { 0%,100% { opacity: .15; } 45% { opacity: 1; } }
@keyframes mo-wave { 0%,100% { transform: scaleY(.12); } 50% { transform: scaleY(.75); } }
@keyframes mo-transmit { from { filter: brightness(.8); } to { filter: brightness(1.18); } }
@keyframes mo-complete { 0% { box-shadow: inset 0 0 0 rgba(var(--mo-accent-rgb), 0); } 35% { box-shadow: inset 0 0 55px rgba(var(--mo-accent-rgb), .25); } 100% { box-shadow: inset 0 0 0 rgba(var(--mo-accent-rgb), 0); } }
body.is-busy .mo-radar .mo-sweep { animation-duration: 1.35s; }
body.is-busy .mo-online::before { animation-duration: .45s; }
button:focus-visible, textarea:focus-visible, input:focus-visible { outline: 2px solid var(--mo-accent) !important; outline-offset: 3px; }

/* 可读性增强：在首轮基础上再放大约 8%，同步增加控件尺寸与留白。 */
.mo-command-main { padding: 30px 34px 28px; }
.mo-eyebrow { font-size: 12px; line-height: 1.45; }
.mo-online { padding: 8px 11px; }
.mo-title { font-size: clamp(33px, 4.4vw, 53px) !important; }
.mo-subtitle { font-size: 16px; line-height: 1.7; }
.mo-radar-label { font-size: 11px; }
.mo-stat { padding: 15px 19px 14px 37px; }
.mo-stat::before { left: 19px; top: 22px; }
.mo-stat small { font-size: 10px; }
.mo-stat strong { font-size: 13px; line-height: 1.45; }
.mo-main { gap: 20px !important; }
.mo-card { padding: 26px 28px 28px !important; }
.mo-section { margin-bottom: 24px; }
.mo-section-index { width: 40px; height: 40px; font-size: 15px; }
.mo-section-text b { font-size: 17px; }
.mo-section-text small { font-size: 11px; }
.mo-section-code { font-size: 11px; }
.mo-input-meta { margin: -2px 2px 10px; font-size: 11px; }
.mo-card .label-wrap, .mo-card button.label-wrap,
.mo-card [data-testid="block-info"], .mo-card .block-title,
.mo-card [data-testid="block-label"], .mo-card [data-testid="block-label"] span { font-size: 14px !important; }
.mo-card .info, .mo-card span.info, .mo-card div.info, .mo-card label > div { font-size: 13px !important; }
.mo-card .wrap { min-height: 50px; }
.mo-card textarea, .mo-card input[type="text"], .mo-card input[type="number"] { font-size: 16px !important; }
#mo-input textarea { min-height: 158px !important; padding: 16px 18px !important; font-size: 17px !important; }
.mo-card .wrap > .head { padding: 7px 16px 5px !important; }
.mo-card input[type="range"] { width: calc(100% - 32px) !important; margin: 6px 16px 13px !important; }
.mo-card .checkbox { font-size: 14px !important; }
button.mo-generate { min-height: 64px !important; font-size: 15px !important; }
.mo-monitor { height: 124px; margin-bottom: 21px; }
.mo-monitor-head { left: 14px; right: 14px; top: 12px; font-size: 10px; }
.mo-wave { inset: 38px 18px 16px; }
.mo-card .mo-readout textarea { font-size: 14px !important; }
#mo-tagged textarea { font-size: 13px !important; }
.mo-hint { margin-top: 18px; padding: 14px 16px; font-size: 13px; }
#mo-examples { margin-top: 20px !important; }
#mo-examples thead th { padding: 13px 16px !important; font-size: 13px !important; }
#mo-examples tbody tr td { padding: 14px 16px !important; font-size: 14px !important; }
#mo-examples tbody tr:hover td:first-child { padding-left: 22px !important; }
.mo-footer { margin-top: 21px; padding-top: 16px; font-size: 11px; }

/* 阵营与语言共用一条控件基线：语言说明移到选择框下方。 */
#mo-faction .container, #mo-language .container {
  display: flex !important;
  flex-direction: column !important;
  height: 100%;
}
#mo-faction .container > .wrap, #mo-language .container > .wrap { order: 2; margin-top: 0 !important; }
#mo-faction .container > [data-testid="block-info"],
#mo-language .container > [data-testid="block-info"] { margin-bottom: 2px !important; }
#mo-language .container > div:not(.wrap) {
  order: 3;
  margin: 7px 0 0 !important;
  color: #78868c !important;
  font-size: 13px !important;
  line-height: 1.55 !important;
}

@media (max-width: 980px) {
  .mo-main { flex-direction: column !important; }
  .mo-main > .column, .mo-main > div { flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }
  .mo-command { grid-template-columns: minmax(0, 1fr) 175px; }
  .mo-radar { flex-basis: 145px; width: 145px; height: 145px; }
}
@media (max-width: 680px) {
  .gradio-container { padding: 12px 12px 22px !important; }
  .mo-command { display: block; min-height: 0; clip-path: polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 14px 100%, 0 calc(100% - 14px)); }
  .mo-command-main { padding: 27px 23px; }
  .mo-radar-wrap { display: none; }
  .mo-title { font-size: 37px !important; }
  .mo-subtitle { font-size: 14px; letter-spacing: .09em; line-height: 1.75; }
  .mo-statusbar { grid-template-columns: 1fr 1fr; }
  .mo-stat:nth-child(2) { border-right: 0; }
  .mo-stat:nth-child(-n+2) { border-bottom: 1px solid var(--mo-line); }
  .mo-card { padding: 22px 18px 23px !important; }
  .mo-stat { padding: 13px 14px 12px 31px; }
  .mo-stat::before { left: 14px; top: 19px; }
  .mo-card > .row, .mo-card .row { flex-direction: column !important; }
  .mo-card > .row > .column, .mo-card .row > .column { min-width: 100% !important; width: 100% !important; }
  .mo-section-code { display: none; }
  #mo-examples { overflow: hidden !important; }
  #mo-examples .table-wrap { overflow-x: auto !important; -webkit-overflow-scrolling: touch; }
  #mo-examples table { min-width: 620px !important; }
  #mo-examples th:nth-child(n+4), #mo-examples td:nth-child(n+4) { display: none; }
  .mo-footer { display: block; }
  .mo-footer span { display: block; margin-bottom: 5px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
}
"""

HERO_HTML = """
<div class="mo-command">
  <div class="mo-command-main">
    <div class="mo-eyebrow">
      <span class="mo-online">BATTLE CONTROL ONLINE</span>
      <span>REDALERT2_ADJUTANT_TTS</span>
    </div>
    <h1 class="mo-title">MENTAL <span>OMEGA</span></h1>
    <p class="mo-subtitle"><b>心灵终结 · 四阵营副官语音指挥终端</b> / TACTICAL VOICE SYNTHESIS</p>
  </div>
  <div class="mo-radar-wrap">
    <svg class="mo-radar" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <radialGradient id="moSweep" gradientUnits="userSpaceOnUse" cx="80" cy="80" r="68">
          <stop offset="0%" stop-color="var(--mo-accent)" stop-opacity=".42"/>
          <stop offset="100%" stop-color="var(--mo-accent)" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <circle cx="80" cy="80" r="70" fill="#080d0f" stroke="rgba(var(--mo-accent-rgb),.28)"/>
      <circle cx="80" cy="80" r="52" fill="none" stroke="rgba(var(--mo-accent-rgb),.18)"/>
      <circle cx="80" cy="80" r="32" fill="none" stroke="rgba(var(--mo-accent-rgb),.16)" stroke-dasharray="2 5"/>
      <path d="M10 80H150M80 10V150M30 30L130 130M130 30L30 130" stroke="rgba(var(--mo-accent-rgb),.09)"/>
      <g class="mo-sweep">
        <path d="M80 80V12A68 68 0 0 1 138.9 46Z" fill="url(#moSweep)"/>
      </g>
      <circle class="mo-blip" cx="111" cy="52" r="2.6" fill="var(--mo-accent)"/>
      <circle class="mo-blip" cx="49" cy="99" r="2.2" fill="var(--mo-accent)"/>
      <circle class="mo-blip" cx="94" cy="119" r="1.8" fill="var(--mo-accent)"/>
      <circle cx="80" cy="80" r="3" fill="var(--mo-accent)"/>
      <path d="M76 4H84M76 156H84M4 76V84M156 76V84" stroke="var(--mo-accent)"/>
    </svg>
    <span class="mo-radar-label">SECTOR SCAN</span>
  </div>
</div>
"""

STATUS_HTML = """
<div class="mo-statusbar">
  <div class="mo-stat"><small>COMMAND UPLINK</small><strong>SECURE / CONNECTED</strong></div>
  <div class="mo-stat"><small>VOICE PROFILES</small><strong>04 FACTIONS READY</strong></div>
  <div class="mo-stat"><small>LANGUAGE PROTOCOL</small><strong>ZH · EN · JA · MIX</strong></div>
  <div class="mo-stat"><small>INFERENCE NODE</small><strong>LOCAL / OFFLINE</strong></div>
</div>
"""

SEC_IN_HTML = """
<div class="mo-section">
  <span class="mo-section-index">01</span>
  <span class="mo-section-text"><b>任务指令</b><small>TRANSMISSION INPUT</small></span>
  <span class="mo-section-line"></span><span class="mo-section-code">CH-01</span>
</div>
"""

INPUT_META_HTML = """
<div class="mo-input-meta"><span id="mo-char-count">0 CHARS</span></div>
"""

SEC_OUT_HTML = """
<div class="mo-section">
  <span class="mo-section-index">02</span>
  <span class="mo-section-text"><b>语音回传</b><small>AUDIO UPLINK</small></span>
  <span class="mo-section-line"></span><span class="mo-section-code">RX-02</span>
</div>
"""

MONITOR_HTML = """
<div class="mo-monitor">
  <div class="mo-monitor-head"><span>NEURAL VOICE WAVEFORM</span><span id="mo-link-state">STANDBY</span></div>
  <div class="mo-wave" aria-hidden="true">
    <i style="--i:0"></i><i style="--i:1"></i><i style="--i:2"></i><i style="--i:3"></i>
    <i style="--i:4"></i><i style="--i:5"></i><i style="--i:6"></i><i style="--i:7"></i>
    <i style="--i:8"></i><i style="--i:9"></i><i style="--i:10"></i><i style="--i:11"></i>
    <i style="--i:12"></i><i style="--i:13"></i><i style="--i:14"></i><i style="--i:15"></i>
    <i style="--i:16"></i><i style="--i:17"></i><i style="--i:18"></i><i style="--i:19"></i>
    <i style="--i:20"></i><i style="--i:21"></i><i style="--i:22"></i><i style="--i:23"></i>
  </div>
</div>
"""

FOOT_HTML = """
<div class="mo-footer">
  <span><b>LANGUAGE TAGS</b> &nbsp;<code>[ZH]</code> <code>[JA]</code> <code>[EN]</code> / MIX AUTO DETECTION</span>
</div>
"""

HINT_HTML = """
<div class="mo-hint">
  <b>通讯提示</b> / 未勾选保存时音频仅保留在临时目录；勾选后写入
  <code>visual/outputs/</code>。若语言标记与文本不匹配，清洗结果可能为空。
</div>
"""


# 纯前端交互：只修改视觉状态，事件参数原样返回，不接触模型和 API 数据。
FORCE_DARK_JS = r"""
() => {
  const factions = { "盟军": "us", "苏联": "su", "焚风": "ff", "厄普西隆": "ep" };
  const setText = (id, value) => {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  };
  window.moSetFaction = (label) => {
    document.body.dataset.faction = factions[label] || factions["盟军"];
  };
  window.moCountText = (text) => {
    setText("mo-char-count", String((text || "").length) + " CHARS");
  };
  window.moSetBusy = () => {
    window.clearTimeout(window.moCompleteTimer);
    document.body.classList.remove("is-complete");
    document.body.classList.add("is-busy");
    setText("mo-link-state", "TRANSMITTING");
    const button = document.querySelector("#mo-generate button") || document.querySelector("#mo-generate");
    if (button) button.textContent = "正在传输 // SYNTHESIZING";
  };
  window.moSettle = (message) => {
    document.body.classList.remove("is-busy");
    document.body.classList.add("is-complete");
    setText("mo-link-state", message && message.indexOf("❌") === 0 ? "LINK ERROR" : "UPLINK READY");
    const button = document.querySelector("#mo-generate button") || document.querySelector("#mo-generate");
    if (button) button.textContent = "GENERATE";
    window.moCompleteTimer = window.setTimeout(() => document.body.classList.remove("is-complete"), 1100);
  };
  try {
    document.documentElement.classList.add("dark");
    document.body.classList.add("dark");
    const url = new URL(window.location.href);
    if (!url.searchParams.get("__theme")) {
      url.searchParams.set("__theme", "dark");
      window.history.replaceState({}, "", url.toString());
    }
  } catch (e) { /* 主题参数不可写不影响服务 */ }
  window.moSetFaction("盟军");
  window.moCountText("Warning. Nuclear silo detected.");
}
"""

FACTION_JS = """(label) => { window.moSetFaction(label); return label; }"""
TEXT_INPUT_JS = """(text) => { window.moCountText(text); return text; }"""
BUSY_JS = """(...args) => { window.moSetBusy(); return args; }"""
SETTLE_JS = """(message) => { window.moSettle(message); }"""


def _build_theme(gr):
    """军用指挥终端基础变量；阵营强调色由 CSS 在浏览器内动态切换。"""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=["Segoe UI", "Microsoft YaHei", "PingFang SC", "system-ui", "sans-serif"],
        font_mono=["Cascadia Mono", "Consolas", "monospace"],
    ).set(
        body_background_fill="#070a0b",
        body_background_fill_dark="#070a0b",
        body_text_color="#e8edf0",
        body_text_color_dark="#e8edf0",
        background_fill_primary="#11171a",
        background_fill_primary_dark="#11171a",
        background_fill_secondary="#0b1012",
        background_fill_secondary_dark="#0b1012",
        block_background_fill="#11171a",
        block_background_fill_dark="#11171a",
        block_border_color="#303b3f",
        block_border_color_dark="#303b3f",
        block_label_text_color="#b7c2c6",
        block_label_text_color_dark="#b7c2c6",
        block_title_text_color="#e0e6e8",
        block_title_text_color_dark="#e0e6e8",
        block_info_text_color="#647278",
        block_info_text_color_dark="#647278",
        block_shadow="none",
        block_shadow_dark="none",
        panel_background_fill="#11171a",
        panel_background_fill_dark="#11171a",
        panel_border_color="#303b3f",
        panel_border_color_dark="#303b3f",
        input_background_fill="#090e10",
        input_background_fill_dark="#090e10",
        input_border_color="#303b3f",
        input_border_color_dark="#303b3f",
        input_border_color_focus="#65bfff",
        input_border_color_focus_dark="#65bfff",
        input_placeholder_color="#4c585d",
        input_placeholder_color_dark="#4c585d",
        color_accent="#65bfff",
        color_accent_soft="#15232b",
        color_accent_soft_dark="#15232b",
        border_color_primary="#293236",
        border_color_primary_dark="#293236",
        border_color_accent="#65bfff",
        border_color_accent_dark="#65bfff",
        link_text_color="#65bfff",
        link_text_color_dark="#65bfff",
        link_text_color_hover="#b9e3ff",
        link_text_color_hover_dark="#b9e3ff",
        button_primary_background_fill="#65bfff",
        button_primary_background_fill_dark="#65bfff",
        button_primary_background_fill_hover="#b9e3ff",
        button_primary_background_fill_hover_dark="#b9e3ff",
        button_primary_text_color="#071012",
        button_primary_text_color_dark="#071012",
        button_primary_text_color_hover="#071012",
        button_primary_text_color_hover_dark="#071012",
        button_primary_border_color="#65bfff",
        button_primary_border_color_dark="#65bfff",
        button_secondary_background_fill="#11191c",
        button_secondary_background_fill_dark="#11191c",
        button_secondary_text_color="#bdc8cb",
        button_secondary_text_color_dark="#bdc8cb",
        button_secondary_border_color="#303b3f",
        button_secondary_border_color_dark="#303b3f",
        slider_color="#65bfff",
        slider_color_dark="#65bfff",
        checkbox_background_color_selected="#65bfff",
        checkbox_background_color_selected_dark="#65bfff",
        checkbox_border_color="#303b3f",
        checkbox_border_color_dark="#303b3f",
        loader_color="#65bfff",
        loader_color_dark="#65bfff",
        table_border_color="#293236",
        table_border_color_dark="#293236",
        table_even_background_fill="#0d1214",
        table_even_background_fill_dark="#0d1214",
        table_odd_background_fill="#0b1012",
        table_odd_background_fill_dark="#0b1012",
    )


# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    with gr.Blocks(
        title="心灵终结副官语音终端",
        theme=_build_theme(gr),
        css=TECH_CSS,
        js=FORCE_DARK_JS,
    ) as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(STATUS_HTML)

        with gr.Row(elem_classes=["mo-main"]):
            with gr.Column(scale=7, elem_classes=["mo-card"]):
                gr.HTML(SEC_IN_HTML)
                txt = gr.Textbox(
                    label="要朗读的文字",
                    lines=4,
                    value="Warning. Nuclear silo detected.",
                    placeholder="输入任意文字；Mix 模式下可用 [ZH]/[JA]/[EN] 标记分段，例如：你好，Commander。[EN]Mission accomplished.[EN]",
                    elem_id="mo-input",
                )
                gr.HTML(INPUT_META_HTML)
                with gr.Row():
                    faction = gr.Dropdown(
                        choices=[f for f, _ in FACTIONS], value="盟军",
                        label="目标阵营", elem_id="mo-faction")
                    language = gr.Dropdown(
                        choices=[l for l, _ in LANGUAGES], value="English（英文）",
                        label="语言",
                        info="单选语言=自动包裹标记；Mix=按标记或自动识别",
                        elem_id="mo-language")
                speed = gr.Slider(0.1, 2.0, value=1.0, step=0.05,
                                  label="语音倍速", info="0.1=极慢  1.0=原速  2.0=极快",
                                  elem_id="mo-speed")
                with gr.Accordion("高级参数", open=False):
                    emotion = gr.Slider(0.1, 1.5, value=0.667, step=0.01,
                                        label="感情变化程度",
                                        info="越大语气起伏越明显，超过 1.0 可能发散")
                    dur_rnd = gr.Slider(0.1, 1.5, value=0.6, step=0.01,
                                        label="音素时长随机度",
                                        info="影响节奏自然度")
                save_file = gr.Checkbox(value=True, label="把生成的音频保存到 visual/outputs/")
                btn = gr.Button("建立语音链路 // GENERATE", variant="primary",
                                elem_id="mo-generate", elem_classes=["mo-generate"])

            with gr.Column(scale=5, elem_classes=["mo-card"]):
                gr.HTML(SEC_OUT_HTML)
                gr.HTML(MONITOR_HTML)
                audio_out = gr.Audio(label="生成结果", type="numpy", autoplay=False,
                                     elem_id="mo-audio")
                status = gr.Textbox(label="状态", interactive=False,
                                    elem_id="mo-status", elem_classes=["mo-readout"])
                tagged_out = gr.Textbox(
                    label="实际送入模型的标注文本", lines=3, interactive=False,
                    elem_id="mo-tagged", elem_classes=["mo-readout"],
                    info="Mix 模式可借此确认语言识别是否正确")
                gr.HTML(HINT_HTML)

        gr.Examples(
            elem_id="mo-examples",
            label="点击载入示例",
            examples=[
                ["Warning. Nuclear silo detected.", "盟军", "English（英文）", 1.0, 0.667, 0.6, True],
                ["Battle control online.", "苏联", "English（英文）", 1.0, 0.667, 0.6, True],
                ["Mission accomplished.", "焚风", "English（英文）", 1.0, 0.667, 0.6, True],
                ["Genetic mutator ready.", "厄普西隆", "English（英文）", 1.0, 0.667, 0.6, True],
                ["你好，指挥官。Mission accomplished.", "盟军", "Mix（多语言混合）", 1.0, 0.667, 0.6, True],
                ["[ZH]核弹发射井已就绪。[ZH][EN]Stand by.[EN]", "苏联", "Mix（多语言混合）", 1.0, 0.667, 0.6, True],
            ],
            inputs=[txt, faction, language, speed, emotion, dur_rnd, save_file],
        )

        # 前端视觉事件：原值直接返回，不增加 Python 回调，也不改变 /tts 签名。
        faction.change(None, faction, None, js=FACTION_JS)
        txt.change(None, txt, None, js=TEXT_INPUT_JS)
        status.change(None, status, None, js=SETTLE_JS)

        # api_name 显式命名：外部调用仍固定使用 /tts，7 个输入、3 个输出不变。
        btn.click(tts_fn,
                  inputs=[txt, faction, language, speed, emotion, dur_rnd, save_file],
                  outputs=[audio_out, status, tagged_out],
                  js=BUSY_JS,
                  api_name="tts")
        # 回车提交走同一条逻辑，但不在 API 里重复暴露。
        txt.submit(tts_fn,
                   inputs=[txt, faction, language, speed, emotion, dur_rnd, save_file],
                   outputs=[audio_out, status, tagged_out],
                   js=BUSY_JS,
                   api_name=False)

        gr.HTML(FOOT_HTML)
    return demo


# ---------------------------------------------------------------------------
def _port_is_free(host, port):
    """判断端口是否可用。

    两个 Windows 上的坑：
      1. 不能给检测用的 socket 设 SO_REUSEADDR —— Windows 的语义是"允许绑定
         已被占用的端口"，会让检测永远返回"空闲"。
      2. 因此先用 connect 探一下有没有人在监听，再补一次真实 bind 兜底。
    """
    import socket

    probe_host = "127.0.0.1" if host in (None, "", "localhost", "0.0.0.0") else host
    bind_host = host if host not in (None, "") else "127.0.0.1"

    # 1) 有进程在监听 -> 一定被占用
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((probe_host, port)) == 0:
                return False
    except OSError:
        pass

    # 2) 真实绑定一次（不设 SO_REUSEADDR）
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def resolve_port(host, port, tries=20):
    """端口被占用时自动往后找。

    这一步很关键：Gradio 4.44 在端口被占用时会**静默退出**（退出码 1，
    不打印任何错误），看上去就像"服务没启动就退了"。所以必须在调用
    launch() 之前自己检查，并给出明确提示。
    """
    if _port_is_free(host, port):
        return port

    print()
    print("!" * 74)
    print("  端口 %d 已被占用（通常是上一次的服务没关干净，或别的程序在用）。" % port)
    for p in range(port + 1, port + tries + 1):
        if _port_is_free(host, p):
            print("  已自动改用端口 %d。" % p)
            print("!" * 74)
            print()
            return p
    raise SystemExit("端口 %d ~ %d 全部被占用，请用 --port 指定其它端口。"
                     % (port, port + tries))


def main():
    global MODEL_PATH, CONFIG_PATH

    ap = argparse.ArgumentParser(description="心灵终结四阵营副官语音合成服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，0.0.0.0 可供局域网访问")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成 Gradio 公网临时链接")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu，默认自动")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    MODEL_PATH, CONFIG_PATH = args.model, args.config

    print("=" * 74)
    print("心灵终结 · 四阵营副官语音合成服务")
    print("=" * 74)
    get_synth(args.device)

    # 再压一次日志级别：build_ui() 里才 import gradio，可能重新动过 logging 配置
    _quiet_logging()

    port = resolve_port(args.host, args.port)
    print("正在启动服务，地址：http://%s:%d"
          % ("127.0.0.1" if args.host == "0.0.0.0" else args.host, port))
    print("（关闭本窗口即停止服务，或按 Ctrl+C）")
    print("-" * 74)

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
