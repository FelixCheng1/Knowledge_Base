import type { Document, ImportTask, Message, QueryResult, Session } from './types'

const base = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8002/api/v1'
const call = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(`${base}${path}`, init)
  if (!response.ok) throw new Error((await response.json().catch(() => ({ detail: response.statusText }))).detail)
  return response.json() as Promise<T>
}

export const api = {
  sessions: () => call<{ items: Session[] }>('/sessions'),
  createSession: () => call<Session>('/sessions', { method: 'POST' }),
  messages: (id: string) => call<{ items: Message[] }>(`/sessions/${id}/messages`),
  documents: (filters?: { status?: string; document_type?: string }) => {
    const params = new URLSearchParams()
    if (filters?.status) params.set('status', filters.status)
    if (filters?.document_type) params.set('document_type', filters.document_type)
    const query = params.toString()
    return call<{ items: Document[] }>(`/documents${query ? `?${query}` : ''}`)
  },
  document: (id: string) => call<Document>(`/documents/${id}`),
  upload: (file: File) => { const form = new FormData(); form.append('file', file); return call<{ document: Document; task: ImportTask }>('/documents', { method: 'POST', body: form }) },
  uploadVersion: (id: string, file: File) => { const form = new FormData(); form.append('file', file); return call<{ document: Document; task: ImportTask }>(`/documents/${id}/versions`, { method: 'POST', body: form }) },
  patchDocument: (id: string, patch: { title?: string; document_type?: string; metadata?: Record<string, unknown> }) => call<Document>(`/documents/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }),
  task: (id: string) => call<ImportTask>(`/import-tasks/${id}`),
  retryTask: (id: string) => call<{ task_id: string; status: string }>(`/import-tasks/${id}/retry`, { method: 'POST' }),
  disable: (id: string) => call<Document>(`/documents/${id}/disable`, { method: 'POST' }),
  ask: (query: string, session_id?: string) => call<QueryResult>('/queries', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, session_id, stream: true }) }),
  result: (id: string) => call<QueryResult>(`/queries/${id}`),
  eventsUrl: (id: string) => `${base}/queries/${id}/events`,
  fileUrl: (id: string, versionId?: string) => `${base}/documents/${id}/file${versionId ? `?version_id=${encodeURIComponent(versionId)}` : ''}`
}
