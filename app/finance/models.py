from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    DISABLED = "disabled"
    INTERRUPTED = "interrupted"


class DocumentType(str, Enum):
    FUND = "fund_product"
    WEALTH = "wealth_management"
    COMPANY_REPORT = "company_report"
    POLICY = "policy"
    EDUCATION = "education_or_faq"
    UNKNOWN = "unknown"


class SourceLocator(BaseModel):
    page: int | None = None
    section: str | None = None
    block_index: int | None = None
    excerpt: str = ""


class FinancialEntity(BaseModel):
    entity_id: str = Field(default_factory=new_id)
    name: str
    entity_type: Literal["product", "share_class", "company", "institution"]
    code: str | None = None
    aliases: list[str] = Field(default_factory=list)
    role: str | None = None


class DocumentVersion(BaseModel):
    version_id: str = Field(default_factory=new_id)
    original_name: str
    stored_path: str
    object_key: str | None = None
    source_path: str | None = None
    checksum: str
    created_at: datetime = Field(default_factory=now)
    publish_date: str | None = None
    report_period: str | None = None
    effective_date: str | None = None
    parser: str = "mineru"
    parse_path: str | None = None
    parse_object_key: str | None = None
    page_count: int | None = None


class FinancialDocument(BaseModel):
    document_id: str = Field(default_factory=new_id)
    title: str
    document_type: DocumentType = DocumentType.UNKNOWN
    document_type_override: DocumentType | None = None
    metadata_overrides: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    active_version_id: str | None = None
    versions: list[DocumentVersion] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    error: str | None = None


class Citation(BaseModel):
    citation_id: str = Field(default_factory=new_id)
    document_id: str
    version_id: str
    title: str
    locator: SourceLocator


class Evidence(BaseModel):
    chunk_id: str
    document_id: str
    version_id: str
    content: str
    title: str
    document_type: str
    locator: SourceLocator
    score: float = 0.0


class ImportTask(BaseModel):
    task_id: str = Field(default_factory=new_id)
    document_id: str
    version_id: str
    status: DocumentStatus = DocumentStatus.PENDING
    stage: str = "等待导入"
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class Session(BaseModel):
    session_id: str = Field(default_factory=new_id)
    title: str = "新对话"
    active_query_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class Message(BaseModel):
    message_id: str = Field(default_factory=new_id)
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    stream: bool = True


class QuestionUnderstanding(BaseModel):
    """问题理解结果：意图分类、消解指代后的独立问题、实体与时间范围。

    由 LLM 结构化输出产生，用于驱动检索过滤与回答策略；
    needs_clarification 为真时其余字段仅供日志参考。
    """
    question_type: Literal["fact", "concept", "summary"] = "fact"
    rewritten_query: str = Field(default="", description="消解指代、补全上下文后的独立问题")
    mentioned_codes: list[str] = Field(default_factory=list, description="用户显式给出的产品/份额/公司代码")
    mentioned_names: list[str] = Field(default_factory=list, description="用户显式给出的产品、公司或文档名称")
    document_type_filter: DocumentType | None = None
    target_document_title: str | None = Field(default=None, description="摘要类问题的目标文档名称")
    time_scope: str | None = Field(default=None, description="限定的时间范围，如 2026Q1")
    needs_clarification: bool = False
    clarification_question: str = ""


class QueryResult(BaseModel):
    query_id: str = Field(default_factory=new_id)
    session_id: str
    status: Literal["processing", "completed", "failed", "clarification_needed"] = "processing"
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class ErrorResponse(BaseModel):
    detail: str


class DocumentListResponse(BaseModel):
    items: list[FinancialDocument]


class SessionListResponse(BaseModel):
    items: list[Session]


class MessageListResponse(BaseModel):
    items: list[Message]


class ImportSubmissionResponse(BaseModel):
    document: FinancialDocument
    task: ImportTask


class RetryImportResponse(BaseModel):
    task_id: str
    status: str


class HealthResponse(BaseModel):
    ok: bool
    ready: bool
    dependencies: dict[str, bool]
