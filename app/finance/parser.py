from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests


def _find_soffice() -> str | None:
    """定位 LibreOffice soffice 可执行文件：环境变量 → 常见安装路径 → PATH。"""
    candidates: list[Path] = []
    env_path = os.getenv("LIBREOFFICE_PATH")
    if env_path:
        candidates.append(Path(env_path))
    for base in (Path(os.getenv("PROGRAMFILES", r"C:\Program Files")), Path(os.getenv("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))):
        candidates.append(base / "LibreOffice" / "program" / "soffice.exe")
    which = shutil.which("soffice")
    if which:
        return which
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def convert_to_pdf(source: Path, output_dir: Path) -> Path:
    """用 LibreOffice 无界面模式把 DOC/DOCX 转为 PDF。

    转换产物保存在 output_dir 下并与原文件同名（扩展名 .pdf），
    调用方负责把转换文件与原文件关联（original_name 仍指向原文件）。
    """
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError("未找到 LibreOffice：请安装 LibreOffice 或设置 LIBREOFFICE_PATH 指向 soffice.exe")
    output_dir.mkdir(parents=True, exist_ok=True)
    # -env:UserInstallation 使用独立用户目录，避免与服务端已开启的 soffice 实例冲突。
    command = [
        soffice, "--headless", "--norestore", "--convert-to", "pdf",
        "--outdir", str(output_dir), f"-env:UserInstallation=file:///{(output_dir / '.lo_profile').as_posix()}",
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    converted = output_dir / f"{source.stem}.pdf"
    if not converted.is_file():
        raise RuntimeError(f"LibreOffice 转换失败：{(completed.stderr or completed.stdout or '').strip()[:400] or '未生成 PDF'}")
    return converted


class MinerUParser:
    """MinerU v4 解析器，PDF、DOC、DOCX 和 Markdown 均走同一金融导入入口。"""

    SUPPORTED = {".pdf", ".doc", ".docx", ".md"}

    def __init__(self, base_url: str, token: str, timeout_seconds: int = 900):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def parse(self, source: Path, output_dir: Path) -> tuple[Path, list[dict]]:
        if source.suffix.lower() not in self.SUPPORTED:
            raise ValueError("仅支持 PDF、DOC、DOCX 和 Markdown 文件")
        if source.suffix.lower() == ".md":
            return source, self._markdown_blocks(source.read_text(encoding="utf-8"))
        if not self.base_url or not self.token:
            raise RuntimeError("MinerU 未配置：请设置 MINERU_BASE_URL 和 MINERU_API_TOKEN")
        # DOCX 首选直传 MinerU（原生解析保留标题/表格结构）；服务端拒绝时回退 LibreOffice 转 PDF。
        # DOC（Word 97-2003 二进制）MinerU 不支持，必须先经 LibreOffice 无界面转换。
        if source.suffix.lower() == ".docx":
            try:
                return self._remote_parse(source, output_dir)
            except RuntimeError as exc:
                if "MinerU 未配置" in str(exc):
                    raise
                converted = convert_to_pdf(source, output_dir / "converted")
                return self._remote_parse(converted, output_dir)
        if source.suffix.lower() == ".doc":
            source = convert_to_pdf(source, output_dir / "converted")
        return self._remote_parse(source, output_dir)

    def _remote_parse(self, source: Path, output_dir: Path) -> tuple[Path, list[dict]]:
        if not self.base_url or not self.token:
            raise RuntimeError("MinerU 未配置：请设置 MINERU_BASE_URL 和 MINERU_API_TOKEN")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{self.base_url}/file-urls/batch",
            json={"files": [{"name": source.name, "data_id": source.stem}], "model_version": "vlm"},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("msg", "MinerU 未接受解析任务"))
        data = payload["data"]
        with requests.Session() as session, source.open("rb") as stream:
            session.trust_env = False
            upload = session.put(data["file_urls"][0], data=stream, timeout=120)
            upload.raise_for_status()

        deadline = time.monotonic() + self.timeout_seconds
        result_url = None
        while time.monotonic() < deadline:
            poll = requests.get(f"{self.base_url}/extract-results/batch/{data['batch_id']}", headers=headers, timeout=30)
            poll.raise_for_status()
            result = poll.json()
            if result.get("code") != 0:
                raise RuntimeError(result.get("msg", "MinerU 解析失败"))
            item = result["data"].get("extract_result", [{}])[0]
            if item.get("state") == "done":
                result_url = item.get("full_zip_url")
                break
            if item.get("state") == "failed":
                raise RuntimeError("MinerU 无法解析该文件")
            time.sleep(3)
        if not result_url:
            raise TimeoutError("MinerU 解析超时")

        output_dir.mkdir(parents=True, exist_ok=True)
        archive = output_dir / "mineru_result.zip"
        # CDN 下载同样绕过系统代理（trust_env=False），代理会截断 OSS 的 TLS 连接。
        with requests.Session() as session:
            session.trust_env = False
            archive.write_bytes(session.get(result_url, timeout=120).content)
        extracted = output_dir / "mineru"
        if extracted.exists():
            shutil.rmtree(extracted)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        markdown = next(extracted.rglob("*.md"), None)
        if markdown is None:
            raise RuntimeError("MinerU 结果中缺少 Markdown")
        blocks = self._load_blocks(extracted, markdown)
        return markdown, blocks

    @staticmethod
    def _markdown_blocks(content: str) -> list[dict]:
        return [{"type": "text", "content": content, "page_idx": None, "section": "全文"}]

    @staticmethod
    def _as_text(value) -> str:
        """MinerU 的 caption/footnote 可能是 str 或 list[str]，统一拉平为字符串。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(part) for part in value if str(part).strip())
        return str(value)

    def _load_blocks(self, extracted: Path, markdown: Path) -> list[dict]:
        content_list = next(extracted.rglob("*_content_list.json"), None)
        if not content_list:
            return self._markdown_blocks(markdown.read_text(encoding="utf-8"))
        raw = json.loads(content_list.read_text(encoding="utf-8"))
        blocks: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text":
                content = self._as_text(item.get("text"))
            elif kind == "table":
                content = "\n".join(filter(None, [
                    self._as_text(item.get("table_caption")),
                    self._as_text(item.get("table_body")),
                    self._as_text(item.get("table_footnote")),
                ]))
            else:
                continue
            if content.strip():
                blocks.append({"type": kind, "content": content, "page_idx": item.get("page_idx"), "section": ""})
        return blocks or self._markdown_blocks(markdown.read_text(encoding="utf-8"))
