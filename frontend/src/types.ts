export type Citation = { citation_id: string; document_id: string; version_id: string; title: string; locator: { page?: number; section?: string; excerpt: string } }
export type Message = { message_id: string; session_id: string; role: 'user' | 'assistant'; content: string; citations: Citation[]; created_at: string }
export type Session = { session_id: string; title: string; updated_at: string; active_query_id?: string | null }
export type DocumentVersion = { version_id: string; original_name: string; checksum: string; created_at: string; publish_date?: string | null; report_period?: string | null; effective_date?: string | null; parser: string; parse_path?: string | null; page_count?: number | null }
export type Document = { document_id: string; title: string; document_type: string; status: string; active_version_id?: string | null; versions: DocumentVersion[]; metadata: Record<string, unknown>; entity_ids: string[]; error?: string | null }
export type ImportTask = { task_id: string; document_id: string; version_id: string; status: string; stage: string; error?: string | null; updated_at: string }
export type QueryResult = { query_id: string; session_id: string; status: string; answer: string; citations: Citation[]; error?: string }
