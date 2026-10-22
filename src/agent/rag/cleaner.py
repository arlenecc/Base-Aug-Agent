"""Text cleaner — removes garbage, tags, and normalizes whitespace."""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from .pandoc_parser import strip_markdown_noise, strip_raw_html

logger = logging.getLogger(__name__)

# 行内 <br> 变体（<br>、<br/>、<br />）——在通用标签剥离之前先替换为空格，
# 否则 "第一行<br>第二行" 剥标签后变成 "第一行第二行"（单词粘连）。
_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)

# 常见命名 HTML 实体（含缺分号的写法，如 "  &nbsp" / "&mdash"）。
# 只匹配已知实体名——任意 `&[a-zA-Z]+` 会把 "R&D labs" 里的 "&D" 误吃掉。
_ENTITY_NAMES = (
    "nbsp|mdash|ndash|amp|lt|gt|quot|apos|hellip|copy|reg|trade|deg|times|divide"
    "|ldquo|rdquo|lsquo|rsquo|laquo|raquo|bull|middot|plusmn|times"
    "|larr|rarr|uarr|darr|harr|lceil|rceil|lfloor|rfloor"
    "|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega"
    "|infin|ne|le|ge|asymp|equiv|sub|sup|cap|cup|int|sum|prod|radic|part"
    "|aacute|eacute|iacute|oacute|uacute|agrave|egrave|agrav|ccedil|uuml|ouml|auml"
    "|szlig|ntilde|atilde|ocirc|acirc|ecirc|icirc|ucirc|yuml"
)
_ENTITY_RE = re.compile(r'&(?:%s);?' % _ENTITY_NAMES, re.IGNORECASE)

# 句末标点：中文句号/问号/叹号 + 英文 . ? !，允许后跟闭引号/闭括号。
# fix_hard_wraps 用它判断一行是否「句子真正结束」——不是句末的换行是
# 硬断行，应与下一行拼回一段。
_SENTENCE_END_RE = re.compile(r'[。！？!?…."][”’」』》）)】\]"\']*$')

# Markdown 结构行：标题/列表/引用/表格/围栏/缩进代码/水平线。
# 这些行的换行是结构性的，fix_hard_wraps 不得把它们与上下行拼接。
_MD_STRUCTURE_RE = re.compile(
    r'^(?:\s{0,3}(?:#{1,6}\s|[-*+]\s|\d{1,9}[.)]\s|>\s?>?|\|'
    r'|```|~~~|(?:-{3,}|\*{3,}|_{3,})\s*$)'
    r'|\s{4,}\S)'
)

# 含竖线的行（Markdown 表格行、_extract_csv 输出的 "a | b | c" 行）一律视为
# 结构行：表格/CSV 的换行是数据边界，拼接会把整张表粘成一行并改变切片边界。
_PIPE_LINE_RE = re.compile(r'\|')


def fix_hard_wraps(text: str) -> str:
    """修复硬断行：非句末换行拼回空格，句末换行保留为段落/句子边界。

    电子书（epub/mobi/html 源）与 OCR 文本普遍存在「每行 ~40 字符的排版
    断行」——句中被 \n 截断，检索切片会在句中分裂。规则（按用户要求）：
      - <br> 变体 → 空格（在拼接前先处理，避免剥标签后单词粘连）；
      - 不以句号/问号/叹号结尾的换行 → 空格（拼回同一段）；
      - 句末换行保留（\n\n 段落边界保持不变）。

    Markdown 结构行（标题 #、列表 -/1.、引用 >、表格 |、代码围栏、缩进
    代码、水平线）的换行是结构性的，原样保留不参与拼接；围栏代码块内部
    逐字保留。
    """
    if not text:
        return ""
    # 硬回车归一：CRLF/CR → LF；不换行空格（pandoc 实体转换残留）→ 空格。
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\u00a0', ' ')
    # 行内 <br> → 空格（围栏内行在循环里逐行处理，不会误伤代码块）。
    # 注意：这里只在非围栏上下文安全，先做全局替换再靠下方围栏逻辑无法
    # 恢复——因此采用逐行扫描，围栏内跳过 _BR_RE。
    lines = text.split('\n')
    out: List[str] = []
    buf: Optional[str] = None  # 正在累积的段落（跨行拼接中）
    in_fence = False
    fence_marker = ""

    def _flush() -> None:
        nonlocal buf
        if buf is not None:
            out.append(buf)
            buf = None

    for line in lines:
        stripped = line.strip()
        # 围栏开关：``` / ~~~（只有同种围栏能闭合）。
        if not in_fence and (stripped.startswith('```') or stripped.startswith('~~~')):
            _flush()
            in_fence = True
            fence_marker = stripped[:3]
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        # 空行 = 段落边界。
        if not stripped:
            _flush()
            out.append('')
            continue

        # Markdown 结构行 / 表格与 CSV 数据行：原样保留，且打断当前段落拼接。
        if _MD_STRUCTURE_RE.match(line) or _MD_STRUCTURE_RE.match(stripped) \
                or _PIPE_LINE_RE.search(stripped):
            _flush()
            out.append(_BR_RE.sub(' ', stripped).strip())
            continue

        # 普通段落行：先清行内 <br>、gfm 硬换行标记（行尾反斜杠）与残留
        # HTML 实体（pandoc 只转换合法实体，畸形实体如 "&mdash" 会原样透传）。
        s = _BR_RE.sub(' ', stripped).strip()
        s = _ENTITY_RE.sub(' ', s)
        s = re.sub(r'\\$', '', s).strip()
        if buf is None:
            buf = s
        else:
            buf = buf + ' ' + s
        if _SENTENCE_END_RE.search(s):
            _flush()

    _flush()
    # 连续 3+ 空行收敛为 1 个空行（段落间距）。
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip()


# HTML 表格块：复杂表格 markdown 无法无损表示，须保留原始 <table> 块而非
# 当普通标签剥掉（否则只剩散落文字、丢失行列结构）。清洗时整体掩成占位符、
# 还原，避免被标签剥离/实体/控制字符等步骤损坏。用私用区字符做占位，避开
# clean_text 里对 \x00 控制字符与各类零宽/空白的删除。
_TABLE_BLOCK_RE = re.compile(r'<table\b[^>]*>[\s\S]*?</table>', re.IGNORECASE)


def _mask_table_blocks(text: str):
    """把 ``<table>...</table>`` 块掩成占位符，返回 (掩码文本, 还原函数)。"""
    blocks: List[str] = []

    def _repl(m: "re.Match") -> str:
        blocks.append(m.group(0))
        return f"\uE000{len(blocks) - 1}\uE000"

    masked = _TABLE_BLOCK_RE.sub(_repl, text)

    def _restore(t: str) -> str:
        for i, b in enumerate(blocks):
            t = t.replace(f"\uE000{i}\uE000", b)
        return t

    return masked, _restore


def strip_html_comments(text: str) -> str:
    """只剥 HTML 注释（``<!-- ... -->``），不动任何 Markdown 语法。

    供 docling/marker/ovis 等「规范 Markdown」路径使用——这些路径在 engine
    里被当作已成型 Markdown，故意不跑 :func:`clean_text`（后者 ``<[^>]+>``
    会误删标题/表格/上标等语法）。但 docling 默认把图片/图题导出成
    ``<!-- image -->`` 占位注释，marker 也会透传少量 HTML 注释，若不单独剥掉
    就会大量进入切片与嵌入。仅移除注释是零风险的。
    """
    if not text:
        return ""
    return re.sub(r'<!--[\s\S]*?-->', ' ', text)


# 围栏代码块起始（允许行首最多 3 空格，CommonMark 规范）。
_FENCE_OPEN_RE = re.compile(r'^\s{0,3}(?:```|~~~)')
# 缩进代码行（4 空格或 1 tab 起头）——内容是字面量，逐字保留。
_INDENT_CODE_LINE_RE = re.compile(r'^(?: {4}|\t)')


def compress_markdown_whitespace(markdown: str) -> str:
    """压缩 Markdown 里不必要的空行与行尾空白——保留格式、提高切片密度。

    引擎产出的「规范 Markdown」（docling/marker/pandoc）常带排版残留：剥掉
    ``<!-- image -->`` 等注释后留下的空行、docling/pandoc 的行尾空格、连续
    多空行。这些空白不改变 Markdown 渲染，却稀释切片与嵌入密度。

    规则（只动「不必要」的，格式一律保留）：
      * **行尾空格/tab 剥掉**——Markdown 行尾空格唯一作用是硬换行，而本管线
        视硬断行为噪音（见 fix_hard_wraps）；围栏/缩进代码内除外（字面量）；
      * **普通行行首 ≤3 空格剥掉**——Markdown 渲染忽略段首 ≤3 空格（再多的
        就是缩进代码），这层空白只可能来自剥标签残留（``<div>正文`` → `` 正文``）
        或排版缩进，属不必要的空格；结构行（标题/列表/引用/表格等）与缩进
        代码行的行首一律不动，保住嵌套列表层级等格式；
      * **连续 2+ 空行收敛为 1 个空行**——1 个空行即段落分隔，多余的不改变
        渲染，只会降低密度；
      * 文档首尾空行剥掉；
      * **围栏代码块**（```/~~~）与**缩进代码行**（4 空格/tab 起头）逐字
        不动——代码里的空行与行尾空白是字面量；
      * HTML ``<table>`` 块整体掩码保护（内部行列结构不动）；
      * 标题/列表/引用/表格等结构行的内容与行首缩进原样保留。

    幂等：压缩一次后再跑零变化。
    """
    if not markdown:
        return ""
    masked, _restore = _mask_table_blocks(markdown)
    out: List[str] = []
    in_fence = False
    fence_marker = ""
    pending_blank = False  # 挂起的空行：连续空行只允许落一个

    def _flush_blank() -> None:
        nonlocal pending_blank
        if pending_blank and out:
            out.append('')
        pending_blank = False

    for line in masked.split('\n'):
        stripped = line.strip()
        if in_fence:
            # 代码块内逐字保留（含空行与行尾空白，全是字面量）。
            out.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if _FENCE_OPEN_RE.match(line):
            _flush_blank()
            in_fence = True
            fence_marker = stripped[:3]
            out.append(line.rstrip())
            continue
        if not stripped:
            pending_blank = True
            continue
        _flush_blank()
        if _INDENT_CODE_LINE_RE.match(line):
            out.append(line)           # 缩进代码行逐字保留
            continue
        if _MD_STRUCTURE_RE.match(line) or _MD_STRUCTURE_RE.match(stripped):
            out.append(line.rstrip())  # 结构行：只剥行尾，行首缩进（嵌套层级）不动
            continue
        # 普通正文行：行首 ≤3 空格（渲染忽略，只可能是剥标签残留）+ 行尾空白
        # 一并剥掉；4+ 行首空格上面已按缩进代码拦走，到不了这里。
        out.append(stripped)
    return _restore('\n'.join(out))


def clean_markdown_document(markdown: str) -> str:
    """对 docling/marker/pandoc 输出的「规范 Markdown」做无损噪音清理。

    这些路径在 engine 里被当作已成型 Markdown，不跑 :func:`clean_text`
    （其 ``<[^>]+>`` 会误删标题/表格/上标等语义语法）。但三个引擎都会往
    markdown 里吐「非正文噪音」，必须统一剥掉，否则大量进入切片与嵌入：

      * **HTML 注释**：docling 默认把图片/图题导出成 ``<!-- image -->``，
        marker/docling 偶尔透传其它 ``<!-- ... -->``（如 ``<!-- page ... -->``、
        ``<!-- formula -->``）。
      * **raw HTML 标签**：pandoc 转 docx/html/epub 会透传 ``<div>`` /
        ``<span class=...>`` / ``<a id=...>`` / ``<sup>`` / ``<table>`` 等；
        marker 也会吐 ``<sup>`` 脚注与原始 HTML；docling 在含 HTML 的源上
        同理。仅剥标签、保留标签内正文，且要求标签首字母开头（避开散文
        ``a < b`` 这类比较运算）。
      * **Markdown 图片语法** ``![...](...)``：pandoc 转 epub/图版 PDF 会
        产出成百上千条（图片本身不入库，留着只污染切片）；其它引擎个别情况
        也会产。
      * **内部锚点链接** ``[文字](#锚点)``：epub 的目录页/交叉引用会产出海量
        这类导航噪音；剥掉链接语法但**保留链接文字**（目录条目名有语义）。

    代码块（围栏/缩进）与行内代码 span 受保护——里面的 ``< >`` / ``![](x)``
    是示例内容，不是噪音。顺序：注释 → raw HTML → 图片 → 内部链接。

    **空白压缩**：最后再过一遍 :func:`compress_markdown_whitespace`——剥掉
    注释/标签后残留的连续空行、行尾空格全部收敛（连续空行 → 1 个，行尾空白
    剥掉），首尾空行去掉；围栏/缩进代码与 ``<table>`` 块逐字不动，标题/列表/
    引用等结构行只剥行尾空白、保留行首缩进。提高切片密度，不改变渲染。

    **表格块保留**：复杂表格 markdown 无法无损表示，整段 ``<table>...</table>``
    会被原样保留（含 ``<tr>/<td>/<th>`` 行列结构，不剥标签），而不是被拆散成
    散落文字。docling/marker 默认把表格导成 markdown ``|`` 语法（本就保留）；
    个别源（HTML 透传、raw HTML 块）产出的 HTML 表格块也借此保住。
    幂等，pandoc 路径在 parser 里已做过同类清理，再跑一次零副作用。
    """
    if not markdown:
        return ""
    md = strip_html_comments(markdown)        # <!-- image --> 等注释
    md = strip_raw_html(md)                   # <div>/<span>/<sup>/<a> 等标签（留正文）
    md = strip_markdown_noise(md)             # ![...](...) 与 [文字](#...)
    md = compress_markdown_whitespace(md)     # 收敛空行/行尾空白（保格式、提密度）
    return md


def clean_text(text: str) -> str:
    """Clean extracted text: remove garbage tags, normalize whitespace,
    strip non-printable characters, and collapse redundant blank lines."""
    if not text:
        return ""
    logger.debug("      清洗文本: 原始 %d 字符", len(text))

    # 0. 硬回车归一：CRLF/CR → LF（后续步骤统一按 \n 处理）。
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 0.25 保护 HTML 表格块：复杂表格 markdown 表达不了，保留原始 <table>
    # 块（含行列结构），待全部清洗步骤跑完后再还原，避免被标签/实体/控制
    # 字符等步骤损坏。
    text, _restore_tables = _mask_table_blocks(text)

    # 0.5 行内 <br> → 空格（必须在通用标签剥离之前，否则单词粘连）。
    text = _BR_RE.sub(' ', text)

    # 1. Remove entire script/style blocks (including their content)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', ' ', text, flags=re.IGNORECASE)

    # 2. Remove HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', ' ', text)

    # 3. Remove remaining HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Remove HTML entities（含缺分号的已知实体名，如 "&nbsp" "&mdash"）
    text = _ENTITY_RE.sub(' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)

    # 5. Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)

    # 6. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # 7. Remove control characters except common whitespace (\n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 8. Replace various Unicode whitespace with normal space（含 \u00a0 NBSP）
    text = re.sub(r'[\u00a0\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)

    # 9. Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]', '', text)

    # 10. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # 10.5 还原 HTML 表格块：标签/实体/控制字符等步骤已跑完，把 <table>
    # 块还原回正文，再交給下方行级垃圾过滤与空行收敛（表格内部行列结构保留）。
    text = _restore_tables(text)

    # 11. Remove lines that are just garbage (very short, no letters/digits)
    lines = text.split('\n')
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        # Skip lines that are mostly non-alphanumeric garbage
        alpha_ratio = sum(1 for c in stripped if c.isalnum() or c.isspace()) / max(len(stripped), 1)
        if alpha_ratio < 0.3 and len(stripped) < 20:
            continue
        # Skip lines that are just page numbers or headers
        if re.match(r'^\d{1,4}$', stripped):
            continue
        if re.match(r'^第\s*\d{1,4}\s*页$', stripped):
            continue
        if re.match(r'^Page\s+\d{1,4}$', stripped, re.IGNORECASE):
            continue
        # Skip separator lines (long runs of the same character)
        if re.match(r'^([\-=_*#~]{3,})\s*$', stripped):
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # 11.5 修复硬断行：非句末换行拼回空格（电子书/OCR 的排版断行）。
    text = fix_hard_wraps(text)

    # 12. Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 13. Strip leading/trailing whitespace
    text = text.strip()

    logger.debug("      清洗文本完成: %d 字符", len(text))
    return text


def normalize_markdown(text: str) -> str:
    """Convert cleaned text to clean markdown format.

    代码块内的空行保留原样，不会被当作段落分隔符。
    非代码块段落之间用一个空行分隔。
    """
    if not text:
        return ""
    logger.debug("      转换 Markdown: %d 字符", len(text))

    lines = text.split('\n')
    result: List[str] = []
    in_code_block = False
    prev_blank = False  # 上一行是否是空行（用于合并连续空行）

    for line in lines:
        stripped = line.strip()

        # Detect code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            result.append(stripped)
            prev_blank = False
            continue

        if in_code_block:
            # 代码块内：保留原始行（包括空行），不做任何修改
            result.append(line)
            prev_blank = False
            continue

        if not stripped:
            # 非代码块的空行：确保段落之间只有一个空行（合并连续空行）
            if not prev_blank and result:
                result.append('')
            prev_blank = True
            continue

        # 非代码块的非空行：strip 后追加
        result.append(stripped)
        prev_blank = False

    # 去除末尾空行
    while result and not result[-1].strip():
        result.pop()

    output = '\n'.join(result)
    logger.debug("      转换 Markdown 完成: %d 行", len(result))
    return output