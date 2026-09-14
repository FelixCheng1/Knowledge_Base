from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import requests


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
        archive.write_bytes(requests.get(result_url, timeout=120).content)
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

    def _load_blocks(self, extracted: Path, markdown: Path) -> list[dict]:
        content_list = next(extracted.rglob("*_content_list.json"), None)
        if not content_list:
            return self._markdown_blocks(markdown.read_text(encoding="utf-8"))
        raw = json.loads(content_list.read_text(encoding="utf-8"))
        blocks: list[dict] = []
        for item in raw:
            kind = item.get("type")
            if kind == "text":
                content = item.get("text", "")
            elif kind == "table":
                content = "\n".join(filter(None, [item.get("table_caption", ""), item.get("table_body", ""), item.get("table_footnote", "")]))
            else:
                continue
            if content.strip():
                blocks.append({"type": kind, "content": content, "page_idx": item.get("page_idx"), "section": ""})
        return blocks or self._markdown_blocks(markdown.read_text(encoding="utf-8"))
