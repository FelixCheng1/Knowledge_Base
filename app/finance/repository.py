from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from pymongo import MongoClient

from app.finance.models import DocumentStatus, FinancialDocument, FinancialEntity, ImportTask, Message, QueryResult, Session


class FinanceRepository:
    """Mongo 持久化层；所有集合均使用 finance_ 前缀，与旧项目隔离。"""

    def __init__(self, mongo_url: str, database: str):
        self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=1500)
        self.db = self.client[database]
        self.documents = self.db.finance_documents
        self.entities = self.db.finance_entities
        self.tasks = self.db.finance_import_tasks
        self.sessions = self.db.finance_sessions
        self.messages = self.db.finance_messages
        self.queries = self.db.finance_queries
        self._indexes_ready = False

    def _ensure_indexes(self) -> None:
        self.documents.create_index("document_id", unique=True)
        self.entities.create_index("entity_id", unique=True)
        self.tasks.create_index("task_id", unique=True)
        self.sessions.create_index("session_id", unique=True)
        self.messages.create_index([("session_id", 1), ("created_at", 1)])
        self.queries.create_index("query_id", unique=True)

    @staticmethod
    def _dump(model: Any) -> dict[str, Any]:
        return model.model_dump(mode="python")

    @staticmethod
    def _clean(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        value.pop("_id", None)
        return value

    def save_document(self, document: FinancialDocument) -> None:
        self.documents.replace_one({"document_id": document.document_id}, self._dump(document), upsert=True)

    def get_document(self, document_id: str) -> FinancialDocument | None:
        raw = self._clean(self.documents.find_one({"document_id": document_id}))
        return FinancialDocument.model_validate(raw) if raw else None

    def list_documents(self, status: str | None = None, document_type: str | None = None) -> list[FinancialDocument]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if document_type:
            query["document_type"] = document_type
        return [FinancialDocument.model_validate(self._clean(item)) for item in self.documents.find(query).sort("updated_at", -1)]

    def update_document(self, document_id: str, changes: dict[str, Any]) -> FinancialDocument | None:
        changes["updated_at"] = datetime.utcnow()
        self.documents.update_one({"document_id": document_id}, {"$set": changes})
        return self.get_document(document_id)

    def save_entity(self, entity: FinancialEntity) -> None:
        self.entities.replace_one({"entity_id": entity.entity_id}, self._dump(entity), upsert=True)

    def find_entities(self, term: str) -> list[FinancialEntity]:
        escaped = {"$regex": re.escape(term), "$options": "i"}
        return [FinancialEntity.model_validate(self._clean(x)) for x in self.entities.find({"$or": [{"name": escaped}, {"code": escaped}, {"aliases": escaped}]})]

    def documents_for_entity_ids(self, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []
        return [item["document_id"] for item in self.documents.find({"entity_ids": {"$in": entity_ids}, "status": DocumentStatus.ACTIVE.value}, {"document_id": 1})]

    def active_version_pairs(self, document_ids: list[str] | None = None, time_scope: str | None = None) -> list[tuple[str, str]]:
        """Return active document/version pairs, optionally constrained by a time scope."""
        query: dict[str, Any] = {
            "status": DocumentStatus.ACTIVE.value,
            "active_version_id": {"$type": "string", "$ne": ""},
        }
        if document_ids:
            query["document_id"] = {"$in": document_ids}
        pairs: list[tuple[str, str]] = []
        for item in self.documents.find(query, {"document_id": 1, "active_version_id": 1, "versions": 1}):
            version_id = item.get("active_version_id")
            if not version_id:
                continue
            if time_scope:
                version = next((v for v in item.get("versions", []) if v.get("version_id") == version_id), {})
                if not self._version_matches_time(version, time_scope):
                    continue
            pairs.append((item["document_id"], version_id))
        return pairs

    @staticmethod
    def _version_matches_time(version: dict[str, Any], time_scope: str) -> bool:
        scope = re.sub(r"\s", "", time_scope).lower()
        text = " ".join(str(version.get(key) or "") for key in ("publish_date", "report_period", "effective_date")).lower()
        if scope in re.sub(r"\s", "", text):
            return True
        year = re.search(r"20\d{2}", scope)
        if not year or year.group(0) not in text:
            return False
        quarter = re.search(r"q([1-4])", scope)
        if quarter:
            markers = ("一", "二", "三", "四")
            return f"{markers[int(quarter.group(1)) - 1]}季度" in text or f"第{quarter.group(1)}季度" in text
        return "年度" in scope and "年度" in text

    def documents_for_title(self, term: str) -> list[str]:
        escaped = {"$regex": re.escape(term), "$options": "i"}
        return [
            item["document_id"]
            for item in self.documents.find({"title": escaped, "status": DocumentStatus.ACTIVE.value}, {"document_id": 1})
        ]

    def save_task(self, task: ImportTask) -> None:
        self.tasks.replace_one({"task_id": task.task_id}, self._dump(task), upsert=True)

    def get_task(self, task_id: str) -> ImportTask | None:
        raw = self._clean(self.tasks.find_one({"task_id": task_id}))
        return ImportTask.model_validate(raw) if raw else None

    def save_session(self, session: Session) -> None:
        self.sessions.replace_one({"session_id": session.session_id}, self._dump(session), upsert=True)

    def get_session(self, session_id: str) -> Session | None:
        raw = self._clean(self.sessions.find_one({"session_id": session_id}))
        return Session.model_validate(raw) if raw else None

    def list_sessions(self) -> list[Session]:
        return [Session.model_validate(self._clean(x)) for x in self.sessions.find().sort("updated_at", -1)]

    def claim_session_query(self, session_id: str, query_id: str) -> bool:
        """Atomically reserve a session for one in-flight query."""
        result = self.sessions.update_one(
            {"session_id": session_id, "$or": [{"active_query_id": None}, {"active_query_id": {"$exists": False}}]},
            {"$set": {"active_query_id": query_id}},
        )
        return result.modified_count == 1

    def release_session_query(self, session_id: str, query_id: str) -> None:
        self.sessions.update_one({"session_id": session_id, "active_query_id": query_id}, {"$set": {"active_query_id": None}})
    def save_message(self, message: Message) -> None:
        self.messages.insert_one(self._dump(message))
        self.sessions.update_one({"session_id": message.session_id}, {"$set": {"updated_at": message.created_at}})

    def list_messages(self, session_id: str) -> list[Message]:
        return [Message.model_validate(self._clean(x)) for x in self.messages.find({"session_id": session_id}).sort("created_at", 1)]

    def delete_session(self, session_id: str) -> bool:
        deleted = self.sessions.delete_one({"session_id": session_id}).deleted_count > 0
        if deleted:
            self.messages.delete_many({"session_id": session_id})
            self.queries.delete_many({"session_id": session_id})
        return deleted

    def save_query(self, result: QueryResult) -> None:
        self.queries.replace_one({"query_id": result.query_id}, self._dump(result), upsert=True)

    def get_query(self, query_id: str) -> QueryResult | None:
        raw = self._clean(self.queries.find_one({"query_id": query_id}))
        return QueryResult.model_validate(raw) if raw else None

    def has_active_query(self, session_id: str) -> bool:
        return self.queries.count_documents({"session_id": session_id, "status": "processing"}) > 0

    def interrupt_processing_queries(self) -> int:
        result = self.queries.update_many({"status": "processing"}, {"$set": {"status": "failed", "error": "服务重启导致查询中断", "updated_at": datetime.utcnow()}})
        return result.modified_count

    def readiness(self) -> bool:
        self.client.admin.command("ping")
        if not self._indexes_ready:
            self._ensure_indexes()
            self._indexes_ready = True
        return True
