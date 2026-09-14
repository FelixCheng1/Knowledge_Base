export type Citation = { citation_id: string; document_id: string; version_id: string; title: string; locator: { page?: number; section?: string; excerpt: string } }
export type Message = { message_id: string; session_id: string; role: 'user' | 'assistant'; content: string; citations: Citation[]; created_at: string }
export type Session = { session_id: string; title: string; updated_at: string }
export type Document = { document_id: string; title: string; document_type: string; status: string; active_version_id?: string; metadata: Record<string, string | null>; error?: string }
export type QueryResult = { query_id: string; session_id: string; status: string; answer: string; citations: Citation[]; error?: string }
