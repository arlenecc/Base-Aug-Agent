"""Web tools: web_scan (fetch + extract text) and webexec_js (browser control)."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - bs4 optional at runtime
    BeautifulSoup = None  # type: ignore

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

# 单次抓取最多读入的字节数。httpx 的非流式 get() 会把整个响应体读进内存，
# 一个几 GB 的文件/流式接口就能把进程打满。
_MAX_FETCH_BYTES = 8 * 1024 * 1024


def _is_ssrf_target(url: str) -> bool:
    """检测典型的 SSRF 目标（云元数据端点 / 链路本地地址）。

    URL 由模型提供，只做协议白名单是不够的：`http://169.254.169.254/latest/
    meta-data/` 会拿到云主机的 IAM 凭据。这里只拦截**字面量** IP——不做 DNS
    解析（会引入阻塞，且对 DNS rebinding 也无能为力），因此是纵深防御的一层，
    不是完整防护。

    有意放行 127.0.0.0/8 与 ::1：本地开发/自测服务是 web_fetch 的正常用途，
    拦掉会让工具在本地场景下不可用。
    """
    try:
        host = (urlparse(url).hostname or "").strip().strip("[]")
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # 非点分十进制字面量的变体：纯十进制整数（http://2852039166/）与
            # 0x 十六进制（http://0xA9FEA9FE/）也会被 getaddrinfo 解析回真实
            # IP——只拦点分形式等于没拦。归一成整数后再判；带点的 host、
            # 超出 v4 范围的大整数按普通主机名放行（DNS 层防御不在本层职责内）。
            if re.fullmatch(r"(?:\d+|0[xX][0-9a-fA-F]+)", host):
                ip = ipaddress.ip_address(int(host, 0))
            else:
                return False
    except ValueError:
        # 主机名（非 IP 字面量）→ 不做判断。
        return False
    if ip.is_loopback or ip.is_global:
        return False
    # 0.0.0.0/8（含 0.0.0.0）：绑定全部接口的通配地址，作为目标会打到本机，
    # 应一并拦截（ipaddress 把 0.0.0.0 判为 unspecified，不在其它分支里）。
    if ip.is_unspecified or (
        ip.version == 4 and ip in ipaddress.ip_network("0.0.0.0/8")
    ):
        return True
    # 链路本地（169.254.0.0/16 含 AWS/GCP/Azure 元数据；fe80::/10）
    if ip.is_link_local:
        return True
    if ip.version == 4:
        # fd00:ec2::254 之外的 IPv4 元数据地址已由 link-local 覆盖。
        return False
    # IPv6 唯一本地地址 fd00::/8 —— 含 fd00:ec2::254（AWS IMDSv6）。
    return ip.is_private


def _block_ssrf_redirect(response):
    """httpx response 钩子：每个重定向 hop 都要重新过 SSRF 校验。

    ``follow_redirects=True`` 时 httpx 会自动跟随 ``Location``，且**不会**对
    重定向目标再调 ``_is_ssrf_target``——攻击者用开放重定向（``302 →
    http://169.254.169.254/latest/meta-data/``）即可绕过输入 URL 的拦截偷到
    云凭据。这里在每个 hop 前校验目标，命中即抛错中断跟随。
    """
    next_req = getattr(response, "next_request", None)
    if next_req is not None and _is_ssrf_target(str(next_req.url)):
        from httpx import HTTPError
        raise HTTPError(
            f"refusing to follow redirect to blocked address: {next_req.url}"
        )


def _resp_ct(resp) -> str:
    """取响应 Content-Type（兼容大小写：真实 httpx 大小写不敏感，测试替身是普通 dict）。"""
    h = getattr(resp, "headers", None) or {}
    return h.get("content-type") or h.get("Content-Type") or ""


def _read_capped(client, url: str, *, extra_headers: Optional[Dict[str, str]] = None,
                 max_bytes: int = _MAX_FETCH_BYTES) -> Tuple[str, str]:
    """GET ``url`` → ``(body_text, content_type)``，body 上限 ``max_bytes``。

    优先走流式读取：``client.get()`` 是非流式的——整个响应体会先完整读进内存，
    才轮到我们截断，对一个 2GB 的直链内存峰值仍然是 2GB。``client.stream()``
    让我们读够上限就关掉连接。测试替身/极简适配器可能只实现了 ``get()``，
    那条路径退化为「get 后截断」。``extra_headers`` 供 AJAX 直取带 XHR 语义头。
    """
    cap = max(1024, int(max_bytes))
    if hasattr(client, "stream"):
        chunks: list = []
        total = 0
        truncated = False
        ctype = ""
        try:
            with client.stream("GET", url, headers=extra_headers or None) as resp:
                if resp.status_code >= 400:
                    return f"__HTTP_ERROR__{resp.status_code}", _resp_ct(resp)
                for block in resp.iter_bytes(chunk_size=65536):
                    if not block:
                        continue
                    remaining = cap - total
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(block) > remaining:
                        chunks.append(block[:remaining])
                        total += remaining
                        truncated = True
                        break
                    chunks.append(block)
                    total += len(block)
                encoding = getattr(resp, "encoding", None) or "utf-8"
                ctype = _resp_ct(resp)
            text = b"".join(chunks).decode(encoding, errors="replace")
            del chunks
            if truncated:
                text += f"\n… [响应超过 {cap // (1024 * 1024)} MB，已截断]"
            return text, ctype
        except AttributeError:
            # 仅当 client 没有真正的 stream 实现（测试替身/残缺实现）才回退。
            pass
        except Exception:
            # 真实传输失败（超时/断流/连接错误）必须向上抛：否则会对同一 URL
            # 再发一次完整请求（最坏 2×30s 且重复下载），且已读内容被整段丢弃。
            raise

    resp = client.get(url)
    try:
        if resp.status_code >= 400:
            return f"__HTTP_ERROR__{resp.status_code}", _resp_ct(resp)
        # 优先按字节裁剪（真实 httpx 的 .content），拿不到字节时退化为
        # .text（更轻量的响应/测试替身只实现 text）。
        raw = getattr(resp, "content", None)
        if isinstance(raw, (bytes, bytearray)):
            truncated = len(raw) > _MAX_FETCH_BYTES
            text = raw[:_MAX_FETCH_BYTES].decode(
                getattr(resp, "encoding", None) or "utf-8", errors="replace")
            del raw
        else:
            text = resp.text or ""
            truncated = len(text) > _MAX_FETCH_BYTES
            text = text[:_MAX_FETCH_BYTES]
    finally:
        resp.close()

    if truncated:
        text += f"\n… [响应超过 {_MAX_FETCH_BYTES // (1024 * 1024)} MB，已截断]"
    return text, _resp_ct(resp)


# ---------------------------------------------------------------------------
# web_scan 内容路由 / 抽取 / 渲染辅助
# ---------------------------------------------------------------------------

# 输出上限（字符）。抓取上限是 _MAX_FETCH_BYTES（8MB），这里限制最终回给模型的文本。
_MAX_OUTPUT = 20000
# 静态递归 iframe 的最大个数（防一个页面挂几十个 iframe 拖垮同步）。
_MAX_IFRAMES = 6
# 抽到的可见文本少于此阈值 → 判定「可能是 JS 动态渲染页（SPA 空壳）」，触发渲染。
_MIN_TEXT_TO_RENDER = 200
# 静态文本低于此值**且**命中动态加载特征 → 也触发渲染。比「纯空壳」宽一档：
# 很多动态站（ASP.NET/AJAX 搜索页、CMS 过滤列表）静态 HTML 里有几千字符的
# 导航/营销文本，正文列表却要 JS 拉取——只看总长度会把它们误判成"静态已够"。
_SOFT_TEXT_TO_RENDER = 4000
# 动态加载特征：AJAX/前端框架/列表过滤库。命中说明正文大概率要执行 JS 才有。
_DYNAMIC_HINT_RE = re.compile(
    r"(?:XMLHttpRequest|\$.ajax\(|\bfetch\s*\(|axios\.[a-z]+|cmsfilter|htmx"
    r"|__doPostBack|kendo|datatables)",
    re.I,
)

# AJAX 端点发现：二次扫描同源 JS 找数据端点并直取（web_scan 零依赖拿动态
# 数据的核心手段之一，对「GET 可得的 JSON API」类站点有效）。
_MAX_AJAX_SCRIPTS = 4      # 二次扫描的同源 JS 文件上限
_MAX_AJAX_SCRIPT_BYTES = 512 * 1024
_MAX_AJAX_ENDPOINTS = 6    # 直取端点上限
# 端点路径：/api/**、/ajax/**、*.ashx/*.asmx（引号包裹的完整字符串字面量——
# 含 { } + $ 等拼接符的运行时模板无法直取，后续过滤）。
_AJAX_ENDPOINT_RE = re.compile(
    "[\"']((?:https?:)?//[^\"']*?/(?:api|ajax)/[A-Za-z0-9_./-]+"
    "|/(?:api|ajax)/[A-Za-z0-9_./-]+"
    "|[^\"']*\\.(?:ashx|asmx))(\\?[^\"']*)?[\"']"
)
_CDN_HOST_RE = re.compile(
    r"(?:googleapis|gstatic|jsdelivr|unpkg|cdn\.|cloudflare|localizecdn"
    r"|bootstrapcdn|jquery)",
    re.I,
)
# 浏览器 UA：不少站点（WAF/CDN）会对非浏览器 UA 返回 403 或挑战页。
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# 本机浏览器渲染（零 Python 依赖）：发现系统已装的 Chromium 系浏览器，
# 用 ``--headless --dump-dom`` 渲染动态页。
# ---------------------------------------------------------------------------
_RENDER_VTIME_BUDGET_MS = 8000   # 虚拟时间预算：等 AJAX 完成
_RENDER_TIMEOUT_S = 45           # 子进程硬超时
_BROWSER_DISCOVERY: Optional[str] = None  # 发现结果缓存（模块级，进程内一次）


def _find_system_browser() -> Optional[str]:
    """找一个可用的 Chromium 系浏览器可执行文件；找不到返回 None。

    顺序：环境变量 ``BASE_AGENT_BROWSER_BIN``（显式指定）→ macOS 应用路径 →
    Windows 常见路径 → PATH（google-chrome/chromium/edge…）。结果进程内缓存
    ——浏览器不会中途消失，省掉每次调用扫盘。
    """
    global _BROWSER_DISCOVERY
    if _BROWSER_DISCOVERY is not None:
        return _BROWSER_DISCOVERY or None
    candidates: List[str] = []
    env_bin = os.environ.get("BASE_AGENT_BROWSER_BIN", "").strip()
    if env_bin:
        candidates.append(env_bin)
    mac = "/Applications/{name}.app/Contents/MacOS/{exe}"
    for name, exe in [
        ("Google Chrome", "Google Chrome"),
        ("Microsoft Edge", "Microsoft Edge"),
        ("Chromium", "Chromium"),
        ("Brave Browser", "Brave Browser"),
        ("Vivaldi", "Vivaldi"),
    ]:
        candidates.append(mac.format(name=name, exe=exe))
    win = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    candidates.append(win)
    candidates += [
        "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
        "microsoft-edge", "brave-browser",
    ]
    found = ""
    for cand in candidates:
        if "/" in cand or "\\" in cand or os.sep in cand:
            if os.path.isfile(cand):
                found = cand
                break
        else:  # PATH 上的命令名
            which = shutil.which(cand)
            if which:
                found = which
                break
    _BROWSER_DISCOVERY = found
    if found:
        logger.info("web_scan 本地渲染使用浏览器: %s", found)
    return found or None

# <script type="application/json"> / ld+json / __NEXT_DATA__：SSR/水合用内嵌数据。
_JSON_SCRIPT_RE = re.compile(
    r"<script[^>]*type=[\"']application/(?:ld\+)?json[\"'][^>]*>(.*?)</script>",
    re.S | re.I,
)
_NEXT_DATA_RE = re.compile(
    r"<script[^>]*id=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>", re.S | re.I,
)
# window.__X__ = {...} / self.FOO = [...]：把状态对象挂到全局（SPA 常见）。
_WINDOW_ASSIGN_RE = re.compile(
    r"(?:window|self|globalThis)\s*\.\s*([A-Za-z_$][\w$]*)\s*=\s*"
)


def _collapse(text: str) -> str:
    """去每行首尾空白、丢空行，压成紧凑文本。"""
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _json_has_data(body: str) -> bool:
    """AJAX 响应是否含实质数据（非错误壳/空结构）。

    判据：JSON 解析成功，且**含非空的数组/嵌套结构**。很多 API 空查询也返回
    200 + 全 null 字段壳（如 applytoeducation 的 GetSearch 无参调用）、错误
    响应只有 Message/ErrorId 标量——这些都不含非空数组，天然被排除。
    """
    try:
        obj = json.loads(body)
    except Exception:
        return False

    def _has_list(o: Any, depth: int = 0) -> bool:
        if depth > 6:
            return False
        if isinstance(o, list):
            return bool(o)
        if isinstance(o, dict):
            return any(_has_list(v, depth + 1) for v in o.values())
        return False

    return _has_list(obj)


def _sniff_kind(text: str, content_type: str) -> str:
    """判断响应是 json / html / xml / text，用于路由（API 数据源 vs 网页）。"""
    ct = (content_type or "").split(";")[0].strip().lower()
    if "json" in ct:
        return "json"
    if "html" in ct:
        return "html"
    if "xml" in ct:
        return "xml"
    head = text.lstrip()[:512]
    if head[:1] in "{[":
        try:
            json.loads(text)
            return "json"
        except Exception:
            pass
    if head.startswith("<?xml"):
        return "xml"
    if head.startswith("<"):
        if re.search(r"<(?:!doctype\s+html|html|head|body)\b", head, re.I):
            return "html"
        return "xml"
    return "text"


def _format_json(text: str, limit: int = _MAX_OUTPUT) -> str:
    """尽量把 JSON 美化输出；解析失败原样返回（都截断到 limit）。"""
    try:
        obj = json.loads(text)
    except Exception:
        return text[:limit]
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) > limit:
        s = s[:limit] + f"\n… [JSON 超过 {limit} 字符，已截断]"
    return s


def _balanced(s: str, i: int) -> Optional[str]:
    """从 ``s[i]``（``{`` 或 ``[``）取到配对的闭合括号，正确处理字符串与转义。"""
    open_c = s[i]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def _extract_embedded_json(html: str) -> List[Tuple[str, Any]]:
    """从 HTML 抽取内嵌结构化数据（API/SSR 状态），返回 ``[(label, obj)]``。

    覆盖：``<script type=application/json>``（含 ld+json）、``__NEXT_DATA__``、
    ``window.__X__={...}`` 全局状态——这些是 SPA「看着空」时真正的数据来源。
    """
    blobs: List[Tuple[str, Any]] = []

    def _try(label: str, raw: str) -> None:
        raw = raw.strip()
        if not raw or len(blobs) >= 4:
            return
        try:
            blobs.append((label, json.loads(raw)))
        except Exception:
            pass

    for m in _JSON_SCRIPT_RE.finditer(html):
        _try("application/json", m.group(1))
    for m in _NEXT_DATA_RE.finditer(html):
        _try("__NEXT_DATA__", m.group(1))
    for m in _WINDOW_ASSIGN_RE.finditer(html):
        j = m.end()
        while j < len(html) and html[j] in " \t":
            j += 1
        if j < len(html) and html[j] in "{[":
            raw = _balanced(html, j)
            if raw:
                _try(m.group(1), raw)
    return blobs


def _extract_html(html: str) -> Dict[str, Any]:
    """从 HTML 抽取可见文本 + 标题/描述 + 内嵌 JSON + iframe/同源脚本 src。"""
    if BeautifulSoup is None:
        return {"title": "", "desc": "", "iframes": [], "json_blobs": [],
                "scripts": re.findall(r"<script[^>]*src=[\"']([^\"']+)[\"']", html, re.I),
                "text": _collapse(re.sub(r"<[^>]+>", " ", html))}
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    desc = ""
    meta = (soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"}))
    if meta and meta.get("content"):
        desc = meta["content"].strip()
    iframes: List[str] = []
    for tag in soup.find_all("iframe"):
        src = (tag.get("src") or "").strip()
        if src and src not in ("about:blank",):
            iframes.append(src)
    scripts: List[str] = []
    for tag in soup.find_all("script", src=True):
        src = (tag.get("src") or "").strip()
        if src:
            scripts.append(src)
    # 先抽内嵌 JSON（脚本还在），再删脚本/style 取可见文本。
    json_blobs = _extract_embedded_json(html)
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return {"title": title, "desc": desc, "iframes": iframes,
            "json_blobs": json_blobs, "scripts": scripts, "text": text}


class WebScanTool(Tool):
    name = "web_scan"
    description = (
        "PREFERRED tool for fetching any web page or online resource. Returns the "
        "page's visible text (HTML/scripts stripped), embedded JSON data, and iframe "
        "content. JS-rendered dynamic pages (SPA, AJAX-loaded lists, search results) "
        "are rendered automatically via the local system browser — no extra setup. "
        "raw=true returns raw HTML instead. Prefer this over curl/wget/code_run "
        "network calls; use webexec_js only when interaction (click/type/submit) is "
        "required."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "raw": {"type": "boolean", "description": "Return raw HTML instead of extracted text (default false)."},
        },
        "required": ["url"],
    }

    config: AgentConfig
    registry: ToolRegistry
    _http: httpx.Client  # shared connection pool (lazy-init per instance)

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self._last_scripts: List[str] = []  # 最近一次抽取的页面脚本 src（AJAX 直取用）
        self._http = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            # 浏览器 UA：默认的 "base-agent/0.1" 会被一部分 WAF/CDN 直接
            # 403/挑战页。安全语义不变（UA 不影响 SSRF 校验）。
            headers={"User-Agent": _BROWSER_UA},
            # 每个重定向 hop 都重新校验目标，防止 SSRF 经开放重定向绕过输入校验。
            event_hooks={"response": [_block_ssrf_redirect]},
        )

    def run(self, url: str, raw: bool = False) -> ToolResult:
        # 协议白名单：url 由模型提供，不加限制的话 file:// / gopher:// 等
        # 可被用来读取本地文件或探测内网。
        if not url.lower().startswith(("http://", "https://")):
            return ToolResult(False, error=f"Only http/https URLs are supported: {url[:80]}")
        # SSRF：拦掉云元数据端点等链路本地地址（见 _is_ssrf_target）。
        if _is_ssrf_target(url):
            return ToolResult(
                False,
                error=f"Refusing to fetch link-local/metadata address: {url[:80]}",
            )
        try:
            text, ctype = _read_capped(self._http, url)
        except Exception as e:
            return ToolResult(False, error=f"Request failed: {e}")
        if text.startswith("__HTTP_ERROR__"):
            return ToolResult(False, error=f"HTTP {text[len('__HTTP_ERROR__'):]}")
        if raw:
            return ToolResult(True, output=text[:_MAX_OUTPUT])

        kind = _sniff_kind(text, ctype)
        # API / JSON / XML / 纯文本数据源：不走 HTML 抽取，直接给数据（美化 JSON）。
        if kind == "json":
            return ToolResult(True, output=_format_json(text)[:_MAX_OUTPUT])
        if kind in ("xml", "text"):
            body = _collapse(text)
            label = "XML" if kind == "xml" else "文本"
            return ToolResult(True, output=f"[{label} 响应]\n{body[:_MAX_OUTPUT]}")

        # ---- HTML：抽可见文本 + 标题/描述 + 内嵌 JSON + iframe ----
        info = _extract_html(text)
        self._last_scripts = info.get("scripts") or []
        visible = _collapse(info["text"])
        rendered: Optional[Dict[str, Any]] = None
        # 可见文本太少 → 疑似 JS 动态渲染的空壳页；或文本不少但命中动态加载
        # 特征（AJAX/CMS 过滤/回传）且总量仍偏小 → 正文列表大概率要执行 JS。
        # 渲染链：browser bridge（如已配置）→ 本机系统浏览器 headless（零依赖，
        # 用已装的 Chrome/Edge 等）。都不可用则退回静态 iframe 递归 + AJAX 直取。
        needs_render = len(visible) < _MIN_TEXT_TO_RENDER or (
            len(visible) < _SOFT_TEXT_TO_RENDER
            and _DYNAMIC_HINT_RE.search(text)
        )
        if needs_render:
            rendered = self._render_via_bridge(url) or self._render_local(url)

        sections: List[str] = []
        head_bits: List[str] = []
        eff_title = info["title"]
        if rendered and rendered.get("title"):
            eff_title = eff_title or str(rendered["title"])
        if eff_title:
            head_bits.append(f"# {eff_title}")
        if info["desc"]:
            head_bits.append(f"> {info['desc']}")

        if rendered is not None:
            rtext = _collapse(str(rendered.get("text", "")))
            if len(rtext) > len(visible):
                visible = rtext
            # 渲染后直接读各 iframe 的文本（跨域 iframe 读不到时文本为空，跳过）。
            for fr in rendered.get("iframes", []) or []:
                if not isinstance(fr, dict):
                    continue
                ft = _collapse(str(fr.get("text", "")))
                if ft:
                    sections.append(f"[iframe: {fr.get('src', '')}]\n{ft}")
        elif info["iframes"]:
            # 无 bridge：静态抓取 iframe src（同源/相对路径已 urljoin 归一）。
            for src, body in self._fetch_iframes(info["iframes"], base=url):
                sections.append(f"[iframe: {src}]\n{body}")

        # 内嵌 JSON（SPA 的 SSR/水合数据、API 内联响应）——「看着空」时真正的信息。
        for label, obj in info["json_blobs"]:
            s = json.dumps(obj, ensure_ascii=False, indent=2)
            if len(s) > 4000:
                s = s[:4000] + "\n… [截断]"
            sections.append(f"[内嵌数据: {label}]\n{s}")

        # AJAX 端点直取（零依赖的动态数据手段）：渲染不可用或渲染后仍偏薄时，
        # 从页面及其同源 JS 里发现数据端点并直接 GET。
        rendered_thin = rendered is not None and len(visible) < _MIN_TEXT_TO_RENDER
        if needs_render and (rendered is None or rendered_thin):
            sections.extend(self._fetch_ajax_endpoints(url, text))

        out_parts: List[str] = []
        if head_bits:
            out_parts.append("\n".join(head_bits))
        if visible.strip():
            out_parts.append(visible)
        out_parts.extend(sections)
        result = "\n\n".join(out_parts).strip()

        # 渲染被请求但不可用（无论是否拿到静态文本）：附上环境探测结果，
        # agent 流程据此决定降级路径（webexec_js / code_run 跑 playwright /
        # 报告用户）。web_scan 自身绝不为此安装任何东西。
        if needs_render and rendered is None:
            env_notes = self._render_env_notes()
            if env_notes:
                result = f"{result}\n\n{env_notes}".strip()

        if not result:
            # 绝不返回空：给出可操作的诊断，说明为什么没抓到、下一步怎么办。
            diag = [f"未能从该页面抽取到正文文本：{url[:80]}"]
            if eff_title:
                diag.append(f"标题: {eff_title}")
            diag.append(f"Content-Type: {ctype or '未知'}")
            if info["iframes"]:
                diag.append(f"检测到 {len(info['iframes'])} 个 iframe: "
                            + ", ".join(info["iframes"][:5]))
            diag.append("页面疑似 JS 动态渲染；已尝试内嵌数据、iframe 与 AJAX 端点直取。")
            result = "\n".join(diag)

        return ToolResult(True, output=result[:_MAX_OUTPUT])

    def _render_env_notes(self) -> str:
        """探测本机渲染能力，返回给 agent 的降级决策说明（多行或空串）。"""
        notes: List[str] = []
        browser = _find_system_browser()
        if browser:
            notes.append(f"本机浏览器：{os.path.basename(browser)}（web_scan 已自动尝试渲染；"
                         "若仍无数据，页面大概率需要登录或交互操作）。")
        else:
            notes.append("本机未找到 Chromium 系浏览器（Chrome/Edge/Chromium/Brave）；"
                         "可安装其一，或设置 BASE_AGENT_BROWSER_BIN 指定路径。")
        try:
            import playwright  # noqa: F401
            notes.append("检测到本机已安装 playwright：可用 code_run 编写 headless "
                         "脚本做登录/点击/提交等交互式抓取（web_scan 自身不使用它）。")
        except ImportError:
            notes.append("本机未安装 playwright；如需交互式抓取（登录/点击/提交），"
                         "可配置 browser_endpoint 后用 webexec_js。")
        return "\n".join(notes)

    def _render_via_bridge(self, url: str) -> Optional[Dict[str, Any]]:
        """用 browser_endpoint 的 Playwright bridge 渲染页面，返回结构化结果或 None。

        返回 ``{"title", "text", "iframes": [{"src","text"}...]}``。桥未配置或不可用
        时返回 None（调用方退回静态策略）。不新增依赖、不新增工具——复用 webexec_js
        已有的 bridge 端点。
        """
        endpoint = getattr(self.config, "browser_endpoint", "") or ""
        if not endpoint:
            return None
        script = (
            "(() => { try {"
            " const fr = Array.from(document.querySelectorAll('iframe')).map(f => ({"
            "   src: f.src || '',"
            "   text: (() => { try {"
            "     const d = f.contentDocument || (f.contentWindow && f.contentWindow.document);"
            "     return d && d.body ? d.body.innerText : '';"
            "   } catch (e) { return ''; } })()"
            " }));"
            " return JSON.stringify({"
            "   title: document.title || '',"
            "   text: document.body ? document.body.innerText : '',"
            "   iframes: fr"
            " });"
            "} catch (e) { return JSON.stringify({ error: String(e) }); } })()"
        )
        try:
            resp = self._http.post(
                endpoint.rstrip("/") + "/exec",
                json={"url": url, "script": script, "action": "",
                      "selector": "", "value": ""},
                timeout=60.0,
            )
        except Exception:
            return None
        try:
            if resp.status_code >= 400:
                return None
            ct = resp.headers.get("content-type", "") if hasattr(resp, "headers") else ""
            data = resp.json() if "json" in ct.lower() else {"result": resp.text}
        except Exception:
            return None
        finally:
            try:
                resp.close()
            except Exception:
                pass
        result = data
        for key in ("result", "value", "output", "data"):
            if isinstance(data, dict) and key in data:
                result = data[key]
                break
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return {"title": "", "text": _collapse(result), "iframes": []}
        return result if isinstance(result, dict) else None

    def _render_local(self, url: str) -> Optional[Dict[str, Any]]:
        """用**本机已安装的浏览器** headless 渲染动态页（零 Python 依赖）。

        browser_endpoint 未配置时的回退：发现系统里的 Chrome/Edge/Chromium/
        Brave，调 ``--headless --dump-dom``（``--virtual-time-budget`` 让页面
        的 AJAX 有时间完成），把渲染后的 DOM 喂回现有静态抽取管线。

        与 bridge 版返回同样的结构。浏览器不存在/启动失败/超时都返回 None
        （调用方退回静态策略 + 诊断信息），绝不抛错拖垮 web_scan。
        每次调用起一个短命浏览器进程——web_scan 频率低（模型按需触发），
        常驻实例的内存代价不值得；进程隔离还天然规避了恶意页面对本进程的影响。
        """
        # 测试/CI 开关（同 AUTOLOAD_SKIP_ENV 模式）：置 1 时绝不启动浏览器。
        if os.environ.get("BASE_AGENT_SKIP_WEB_RENDER"):
            return None
        exe = _find_system_browser()
        if not exe:
            return None
        try:
            proc = subprocess.run(
                [
                    exe,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-agent={_BROWSER_UA}",
                    # 虚拟时间预算：把页面的 setTimeout/XHR 快进到完成——
                    # 这是 dump-dom 能拿到 AJAX 列表的关键。
                    f"--virtual-time-budget={_RENDER_VTIME_BUDGET_MS}",
                    "--timeout=30000",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                timeout=_RENDER_TIMEOUT_S,
            )
        except Exception as e:  # 浏览器启动失败 / 超时——渲染失败不是错误
            logger.warning("本地浏览器渲染失败(%s): %s", url[:80], e)
            return None
        html = (proc.stdout or b"").decode("utf-8", errors="replace")
        if not html.strip() or not re.search(r"<(?:!doctype\s+html|html)[\s>]", html[:512], re.I):
            logger.warning("本地浏览器渲染无有效 DOM(%s)", url[:80])
            return None
        info = _extract_html(html)
        return {
            "title": info["title"],
            "text": _collapse(info["text"]),
            "iframes": [{"src": src, "text": ""} for src in info["iframes"]],
        }

    def _fetch_ajax_endpoints(self, page_url: str, page_html: str) -> List[str]:
        """AJAX 端点发现与直取：从页面 HTML 和同源 JS 里找数据端点并 GET。

        web_scan 零依赖拿动态数据的手段之一（对「GET 可得的 JSON API」站点
        有效）。流程：
          1. 收集页面引用的**同源**脚本（排除 CDN，≤ _MAX_AJAX_SCRIPTS 个）；
          2. 在 HTML 与这些 JS 里正则发现端点字面量（/api/**、/ajax/**、
             *.ashx/*.asmx；含 ``{}+$`` 等拼接符的运行时模板无法直取，跳过）；
          3. 逐个 GET（带 X-Requested-With/Referer，复用 client 的 cookies），
             响应为 JSON 且含实质数据（非空数组/嵌套对象）才输出。

        返回 ``["[AJAX 数据: url]\\n{...}", ...]`` 段落（每段截 4000 字符）。
        """
        parts = urlparse(page_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        seen: set = set()
        out: List[str] = []
        tried = 0

        def _try_endpoint(ep: str) -> None:
            nonlocal tried
            ep = urljoin(page_url, ep)
            if ep in seen or tried >= _MAX_AJAX_ENDPOINTS:
                return
            if not ep.lower().startswith(("http://", "https://")):
                return
            if _is_ssrf_target(ep) or not ep.startswith(origin):
                return  # 只直取同源端点：跨域 API 未必允许跨站调用，且有 SSRF 面
            seen.add(ep)
            tried += 1
            try:
                body, ct = _read_capped(self._http, ep, extra_headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": page_url,
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                })
            except Exception:
                return
            if body.startswith("__HTTP_ERROR__"):
                return
            if _sniff_kind(body, ct) != "json":
                return
            s = _format_json(body, limit=4000)
            if not _json_has_data(body):
                return
            logger.info("web_scan AJAX 直取命中: %s", ep[:100])
            out.append(f"[AJAX 数据: {ep}]\n{s}")

        # 1) 页面 HTML 本身（内联脚本常直接写死端点）
        for m in _AJAX_ENDPOINT_RE.finditer(page_html):
            ep = m.group(1) + (m.group(2) or "")
            if not any(c in ep for c in "{}+$<"):
                _try_endpoint(ep)
        # 2) 同源静态 JS 二次扫描（端点常在打包后的 JS 里）
        scripts = []
        for src in (self._last_scripts or [])[: _MAX_AJAX_SCRIPTS + 4]:
            abs_src = urljoin(page_url, src)
            if _CDN_HOST_RE.search(abs_src):
                continue
            if not abs_src.startswith(origin):
                continue
            scripts.append(abs_src)
        for js_url in scripts[:_MAX_AJAX_SCRIPTS]:
            if tried >= _MAX_AJAX_ENDPOINTS and out:
                break
            try:
                js_body, js_ct = _read_capped(
                    self._http, js_url, max_bytes=_MAX_AJAX_SCRIPT_BYTES)
            except Exception:
                continue
            if js_body.startswith("__HTTP_ERROR__"):
                continue
            for m in _AJAX_ENDPOINT_RE.finditer(js_body):
                ep = m.group(1) + (m.group(2) or "")
                if any(c in ep for c in "{}+$<"):
                    continue
                _try_endpoint(ep)
        return out

    def _fetch_iframes(self, srcs: List[str], base: str) -> List[Tuple[str, str]]:
        """静态抓取 iframe（urljoin 归一 + 协议/SSRF 校验 + 抽取文本），有界。"""
        out: List[Tuple[str, str]] = []
        for src in srcs[:_MAX_IFRAMES]:
            abs_src = urljoin(base, src)
            if not abs_src.lower().startswith(("http://", "https://")):
                continue
            if _is_ssrf_target(abs_src):
                continue
            try:
                t, ct = _read_capped(self._http, abs_src)
            except Exception:
                continue
            if t.startswith("__HTTP_ERROR__"):
                continue
            if _sniff_kind(t, ct) == "html":
                body = _collapse(_extract_html(t)["text"])
            else:
                body = _collapse(t)
            if body.strip():
                out.append((abs_src, body))
        return out


class WebExecJsTool(Tool):
    name = "webexec_js"
    description = (
        "Control a browser via a Playwright-style bridge: navigate to a URL and "
        "evaluate JavaScript. Requires the agent's browser_endpoint to be set; "
        "otherwise returns guidance on enabling it."
    )
    # Browser-scoped: does not touch the local filesystem or system. The agent
    # is expected to ask the user (via ask_user) before irreversible web actions.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to (optional if only evaluating)."},
            "script": {"type": "string", "description": "JavaScript to evaluate in the page."},
            "action": {"type": "string", "description": "Optional action: 'click','type','screenshot'."},
            "selector": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["script"],
    }

    config: AgentConfig
    registry: ToolRegistry
    _http: httpx.Client  # shared connection pool (lazy-init per instance)

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self._http = httpx.Client(timeout=60.0)

    def run(self, script: str, url: str = "", action: str = "",
            selector: str = "", value: str = "") -> ToolResult:
        endpoint = getattr(self.config, "browser_endpoint", "") or ""
        if not endpoint:
            return ToolResult(False, error=(
                "Browser bridge not configured. Set config.browser_endpoint to a "
                "Playwright HTTP bridge (e.g. http://localhost:9222) to use webexec_js."
            ))
        payload = {"url": url, "script": script, "action": action,
                   "selector": selector, "value": value}
        try:
            resp = self._http.post(endpoint.rstrip("/") + "/exec", json=payload)
        except Exception as e:
            return ToolResult(False, error=f"Browser bridge error: {e}")
        try:
            if resp.status_code >= 400:
                return ToolResult(False, error=f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"result": resp.text}
        finally:
            resp.close()
        return ToolResult(True, output=json.dumps(data, ensure_ascii=False)[:8000])
