"""Web tools: web_scan (fetch + extract text) and webexec_js (browser control)."""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - bs4 optional at runtime
    BeautifulSoup = None  # type: ignore

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


class WebScanTool(Tool):
    name = "web_scan"
    description = "Fetch a URL and return its visible text content (HTML stripped, scripts removed)."
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
        self._http = httpx.Client(timeout=30.0, follow_redirects=True,
                                  headers={"User-Agent": "base-agent/0.1"})

    def run(self, url: str, raw: bool = False) -> ToolResult:
        try:
            resp = self._http.get(url)
        except Exception as e:
            return ToolResult(False, error=f"Request failed: {e}")
        # Use with-statement to ensure the response body stream is closed
        # even when an exception occurs mid-processing.
        with resp:
            if resp.status_code >= 400:
                return ToolResult(False, error=f"HTTP {resp.status_code}")
            text = resp.text
        if raw:
            return ToolResult(True, output=text[:20000])
        if BeautifulSoup is not None:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            content = soup.get_text(separator="\n")
        else:
            # very rough fallback
            import re

            content = re.sub(r"<script.*?</script>", "", text, flags=re.S | re.I)
            content = re.sub(r"<style.*?</style>", "", content, flags=re.S | re.I)
            content = re.sub(r"<[^>]+>", " ", content)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return ToolResult(True, output="\n".join(lines)[:20000])


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
        with resp:
            if resp.status_code >= 400:
                return ToolResult(False, error=f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"result": resp.text}
        return ToolResult(True, output=json.dumps(data, ensure_ascii=False)[:8000])
