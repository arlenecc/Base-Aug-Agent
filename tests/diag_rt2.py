"""验证双视图历史方案的前提：raw_content 能否逐 token 复现上一轮序列。

思路：assistant 历史消息额外带 raw_content（模型原始生成文本），本地后端
在送模板前把它重组为「content = 开标签 + raw，reasoning_content = 字符串」，
阻止模板自动拆分，使历史渲染与上一轮生成序列完全一致。

本脚本用真实 tokenizer + 真实模板验证前缀是否逐 token 对齐。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "qwen3.5-4b-agentic-coder-v4-4bit-mlx",
)

LT, GT = chr(60), chr(62)
THO = LT + "think" + GT
THE = LT + "/think" + GT

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model, tok = load(MODEL_DIR, lazy=False)
sampler = make_sampler(temp=0.0)

from agent.config import AgentConfig
from agent.tools import ToolRegistry


class CB:
    def __getattr__(self, n):
        return lambda *a, **k: None


cfg = AgentConfig(workspace=os.getcwd())
reg = ToolRegistry(config=cfg, callbacks=CB())
schemas = reg.schemas()


def render(messages):
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=schemas, enable_thinking=True,
    )


Q1 = "帮我用 Python 写一个快速排序"
Q2 = "再加上单元测试"


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---------- 第一轮 ----------
p1 = render([{"role": "user", "content": Q1}])
ids1 = tok.encode(p1)
gen_ids, raw_text = [], ""
for r in stream_generate(model, tok, p1, max_tokens=150, sampler=sampler):
    gen_ids.append(r.token)
    raw_text += r.text
print("第一轮: prompt %d tok, 生成 %d tok" % (len(ids1), len(gen_ids)))
print("原始输出前 60 字:", repr(raw_text[:60]))

cached = ids1 + gen_ids


def try_variant(label, build_msg):
    msgs2 = [
        {"role": "user", "content": Q1},
        build_msg(raw_text),
        {"role": "user", "content": Q2},
    ]
    p2 = render(msgs2)
    ids2 = tok.encode(p2)
    n = common_prefix_len(ids2, cached)
    ok = n == len(cached)
    print("%-46s 公共前缀 %5d / 缓存 %5d  -> %s"
          % (label, n, len(cached), "完全对齐 ✅" if ok else "不一致 ❌"))
    if not ok:
        print("      分歧处缓存侧:", repr(tok.decode(cached[max(0, n - 25):n + 25])))
        print("      分歧处新prompt:", repr(tok.decode(ids2[max(0, n - 25):n + 25])))
    return ok


print()
print("=== 变体对照 ===")
# A: 当前做法（解析后正文，无原始）
try_variant("A 现状：content=解析后正文", lambda raw: {
    "role": "assistant", "content": "这是一段解析后的回复正文。",
})
# B: content=原始文本（不带开标签）
try_variant("B content=raw（不带开标签）", lambda raw: {
    "role": "assistant", "content": raw,
})
# C: content=开标签+raw，reasoning_content=""（阻止自动拆分）
try_variant("C content=开标签+raw, reasoning_content=''", lambda raw: {
    "role": "assistant", "content": THO + "\n" + raw, "reasoning_content": "",
})
# D: 同 C 但 reasoning_content 为非空字符串
try_variant("D content=开标签+raw, reasoning_content='x'", lambda raw: {
    "role": "assistant", "content": THO + "\n" + raw, "reasoning_content": "x",
})
