import { useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import {
  Alert,
  App as AntApp,
  Avatar,
  Badge,
  Breadcrumb,
  Button,
  Card,
  ConfigProvider,
  Divider,
  Empty,
  Input,
  Layout,
  List,
  Menu,
  Modal,
  Select,
  Space,
  Spin,
  Statistic,
  Tag,
  Tooltip,
  Typography,
  Upload,
  message,
} from 'antd'
import {
  ArrowRightOutlined,
  BookOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloudUploadOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  FileSearchOutlined,
  FileTextOutlined,
  FilterOutlined,
  FolderOpenOutlined,
  LeftOutlined,
  MessageOutlined,
  PauseCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  RightOutlined,
  SafetyCertificateOutlined,
  SearchOutlined,
  SendOutlined,
  UploadOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy } from 'pdfjs-dist'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { api } from './api'
import type { Citation, Document, ImportTask, Message, QueryResult, Session } from './types'
import './styles.css'

GlobalWorkerOptions.workerSrc = pdfWorkerUrl

const { Header, Sider, Content } = Layout
const { Text, Title, Paragraph } = Typography

type PreviewState = { document: Document; versionId: string; page: number }

type StatusMeta = { label: string; color: 'green' | 'gold' | 'red' | 'blue' | 'default'; icon: React.ReactNode }

const STATUS_META: Record<string, StatusMeta> = {
  active: { label: '已启用', color: 'green', icon: <CheckCircleOutlined /> },
  processing: { label: '处理中', color: 'blue', icon: <ClockCircleOutlined /> },
  pending: { label: '等待处理', color: 'gold', icon: <ClockCircleOutlined /> },
  failed: { label: '处理失败', color: 'red', icon: <CloseCircleOutlined /> },
  interrupted: { label: '已中断', color: 'gold', icon: <WarningOutlined /> },
  disabled: { label: '已停用', color: 'default', icon: <PauseCircleOutlined /> },
}

const DOCUMENT_TYPES = [
  { value: 'fund_product', label: '基金产品' },
  { value: 'wealth_management', label: '理财与风险揭示' },
  { value: 'company_report', label: '公司报告' },
  { value: 'policy', label: '政策法规' },
  { value: 'education_or_faq', label: '投教与 FAQ' },
  { value: 'unknown', label: '待确认' },
]

const TERMINAL_TASK_STATES = new Set(['active', 'failed', 'disabled', 'interrupted'])

function statusMeta(status: string): StatusMeta {
  return STATUS_META[status] ?? { label: status, color: 'default', icon: <ClockCircleOutlined /> }
}

function documentTypeLabel(type: string) {
  return DOCUMENT_TYPES.find(item => item.value === type)?.label ?? type
}

function formatTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function formatSessionTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

function displaySessionTitle(session: Session) {
  const title = session.title?.trim()
  return title && title !== '新对话' ? title : `新对话 · ${session.session_id.slice(0, 6)}`
}

function latestMessageCitations(items: Message[]) {
  const latestAssistant = [...items].reverse().find(item => item.role === 'assistant')
  return latestAssistant?.citations ?? []
}

function AppShell() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const { pathname, search } = useLocation()
  const navigate = useNavigate()
  const refresh = useCallback(async () => {
    const [sessionResult, documentResult] = await Promise.all([api.sessions(), api.documents()])
    setSessions(sessionResult.items)
    setDocuments(documentResult.items)
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const selectedSessionId = new URLSearchParams(search).get('session')
  const navItems = [
    { key: 'chat', icon: <MessageOutlined />, label: <Link to="/">问答工作台</Link> },
    { key: 'documents', icon: <DatabaseOutlined />, label: <Link to="/documents">金融资料库</Link> },
  ]

  return <Layout className="app-frame">
    <Sider width={252} theme="light" className="app-sider">
      <div className="brand-lockup">
        <div className="brand-mark"><BookOutlined /></div>
        <div>
          <div className="brand-name">掌柜智库</div>
          <div className="brand-caption">FINANCE DESK</div>
        </div>
      </div>
      <div className="sider-action">
        <Link to="/"><Button type="primary" icon={<PlusOutlined />} block>新建对话</Button></Link>
      </div>
      <Menu className="main-menu" mode="inline" selectedKeys={pathname.startsWith('/documents') ? ['documents'] : ['chat']} items={navItems} />
      <div className="side-section-head">
        <Text className="side-section-title">近期会话</Text>
        <Badge count={sessions.length} overflowCount={99} className="session-count" />
      </div>
      <div className="session-list">
        <List
          size="small"
          dataSource={sessions}
          locale={{ emptyText: <span className="muted-empty">暂无会话</span> }}
          renderItem={item => {
            const title = displaySessionTitle(item)
            return <List.Item className={`session-item ${selectedSessionId === item.session_id ? 'selected' : ''}`}>
              <Link to={`/?session=${item.session_id}`} className="session-link" title={title}>
                <MessageOutlined />
                <span className="session-label"><span>{title}</span><small>{formatSessionTime(item.updated_at)}</small></span>
              </Link>
              <Tooltip title="删除会话">
                <Button
                  aria-label={`删除会话 ${title}`}
                className="session-delete"
                type="text"
                size="small"
                icon={<DeleteOutlined />}
                  onClick={() => api.deleteSession(item.session_id).then(async () => {
                    if (selectedSessionId === item.session_id) navigate('/')
                    await refresh()
                  }).catch(err => message.error(err instanceof Error ? err.message : '删除失败'))}
                />
              </Tooltip>
            </List.Item>
          }}
        />
      </div>
      <div className="sider-footer">
        <Badge status="success" />
        <span>本地知识库已连接</span>
        <Tag bordered={false}>金融版</Tag>
      </div>
    </Sider>
    <Layout className="main-layout">
      <Header className="topbar">
        <div className="topbar-mobile-brand"><div className="brand-mark small"><BookOutlined /></div><Text strong>掌柜智库</Text></div>
        <Space className="mobile-nav" size={6}>
          <Link to="/"><Button size="small" type={pathname === '/' ? 'primary' : 'text'} icon={<MessageOutlined />}>问答</Button></Link>
          <Link to="/documents"><Button size="small" type={pathname.startsWith('/documents') ? 'primary' : 'text'} icon={<DatabaseOutlined />}>资料</Button></Link>
        </Space>
        <div className="topbar-status"><Badge status="success" text="服务在线" /><Divider type="vertical" /><Text type="secondary">本地资料 · 不联网</Text></div>
      </Header>
      <Content className="page-content">
        <Routes>
          <Route path="/" element={<Chat documents={documents} sessions={sessions} onRefresh={refresh} />} />
          <Route path="/documents" element={<Documents documents={documents} onRefresh={refresh} />} />
        </Routes>
      </Content>
    </Layout>
  </Layout>
}

function PdfPreview({ src, initialPage }: { src: string; initialPage: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy>()
  const [page, setPage] = useState(Math.max(1, initialPage))
  const [pageCount, setPageCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [rendering, setRendering] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => setPage(Math.max(1, initialPage)), [initialPage, src])
  useEffect(() => {
    let cancelled = false
    let loadedPdf: PDFDocumentProxy | undefined
    const controller = new AbortController()
    setPdf(undefined); setPageCount(0); setLoading(true); setError('')
    void fetch(src, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error(`原文加载失败（HTTP ${response.status}）`); return response.arrayBuffer() })
      .then(data => getDocument({ data }).promise)
      .then(result => {
        loadedPdf = result
        if (cancelled) { void result.destroy(); return }
        setPdf(result); setPageCount(result.numPages); setPage(current => Math.min(Math.max(1, initialPage), result.numPages))
      })
      .catch(err => { if (!cancelled && err?.name !== 'AbortError') setError(err instanceof Error ? err.message : 'PDF 原文加载失败') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true; controller.abort(); if (loadedPdf) void loadedPdf.destroy() }
  }, [src, initialPage])
  useEffect(() => {
    if (!pdf || !canvasRef.current || page < 1 || page > pdf.numPages) return
    let cancelled = false
    setRendering(true)
    void pdf.getPage(page).then(pdfPage => {
      if (cancelled || !canvasRef.current) return
      const viewport = pdfPage.getViewport({ scale: 1.35 })
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')
      if (!context) throw new Error('浏览器不支持 Canvas 预览')
      canvas.width = viewport.width; canvas.height = viewport.height
      canvas.style.aspectRatio = `${viewport.width} / ${viewport.height}`
      return pdfPage.render({ canvasContext: context, viewport }).promise
    }).catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : 'PDF 页面渲染失败') })
      .finally(() => { if (!cancelled) setRendering(false) })
    return () => { cancelled = true }
  }, [pdf, page])

  if (loading) return <div className="pdf-state"><Spin /> <Text type="secondary">正在加载原文…</Text></div>
  if (error) return <Alert type="error" showIcon message="原文预览失败" description={error} />
  if (!pdf || !pageCount) return <Empty description="没有可预览的 PDF 页面" />
  return <div className="pdf-viewer">
    <div className="pdf-toolbar">
      <Space size={8}><Button size="small" icon={<LeftOutlined />} disabled={page <= 1 || rendering} onClick={() => setPage(value => Math.max(1, value - 1))}>上一页</Button><Text className="page-counter">第 {page} / {pageCount} 页</Text><Button size="small" icon={<RightOutlined />} disabled={page >= pageCount || rendering} onClick={() => setPage(value => Math.min(pageCount, value + 1))}>下一页</Button></Space>
      {rendering && <Text type="secondary">正在渲染…</Text>}
    </div>
    <div className="pdf-canvas-wrap"><canvas ref={canvasRef} aria-label={`PDF 第 ${page} 页`} /></div>
  </div>
}

function Chat({ documents, sessions, onRefresh }: { documents: Document[]; sessions: Session[]; onRefresh: () => Promise<void> }) {
  const [sessionId, setSessionId] = useState<string>()
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [preview, setPreview] = useState<PreviewState>()
  const pendingAssistantRef = useRef<Map<string, Message>>(new Map())
  const recoveringQueriesRef = useRef<Set<string>>(new Set())
  const activeSessionRef = useRef<string | undefined>(undefined)
  const navigate = useNavigate()
  const { search } = useLocation()
  const selectedSessionId = new URLSearchParams(search).get('session')

  const resumeQuery = useCallback(async (queryId: string, targetSessionId: string) => {
    if (recoveringQueriesRef.current.has(queryId)) return
    recoveringQueriesRef.current.add(queryId)
    const assistantId = `pending-${queryId}`
    const pendingAssistant: Message = {
      message_id: assistantId, session_id: targetSessionId, role: 'assistant', content: '', citations: [],
      created_at: new Date().toISOString(),
    }
    pendingAssistantRef.current.set(targetSessionId, pendingAssistant)
    setMessages(old => old.some(item => item.message_id === assistantId) ? old : [...old, pendingAssistant])
    setLoading(true)
    setProgress('恢复查询')
    try {
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const result = await api.result(queryId)
        if (result.status !== 'processing') {
          pendingAssistantRef.current.delete(targetSessionId)
          if (activeSessionRef.current === targetSessionId) {
            setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: result.answer || result.error || '查询失败', citations: result.citations ?? [] } : item))
            setCitations(result.citations ?? [])
            setLoading(false)
            setProgress('')
          }
          void onRefresh()
          return
        }
        await new Promise(resolve => window.setTimeout(resolve, 1000))
      }
      if (activeSessionRef.current === targetSessionId) {
        setLoading(false)
        setProgress('')
        message.warning('查询仍在后台处理，可稍后从会话历史查看结果')
      }
    } catch {
      if (activeSessionRef.current === targetSessionId) {
        setLoading(false)
        setProgress('')
      }
    } finally {
      recoveringQueriesRef.current.delete(queryId)
    }
  }, [onRefresh])

  useEffect(() => {
    let cancelled = false
    activeSessionRef.current = selectedSessionId ?? undefined
    if (!selectedSessionId) { setSessionId(undefined); setMessages([]); setCitations([]); return () => { cancelled = true } }
    setSessionId(selectedSessionId); setCitations([])
    void api.messages(selectedSessionId).then(result => {
      if (cancelled) return
      const pendingAssistant = pendingAssistantRef.current.get(selectedSessionId)
      const merged = pendingAssistant
        ? [...result.items, pendingAssistant].sort((left, right) => left.created_at.localeCompare(right.created_at))
        : result.items
      setMessages(merged)
      setCitations(latestMessageCitations(merged))
      const activeQueryId = sessions.find(item => item.session_id === selectedSessionId)?.active_query_id
      if (activeQueryId && !pendingAssistant) void resumeQuery(activeQueryId, selectedSessionId)
    }).catch(() => {
      if (!cancelled) { setMessages([]); setCitations([]) }
    })
    return () => { cancelled = true }
  }, [selectedSessionId, sessions, resumeQuery])

  const ask = async () => {
    const content = question.trim()
    if (!content || loading) return
    setQuestion(''); setLoading(true); setCitations([]); setProgress('理解问题')
    const userMessageId = `pending-user-${crypto.randomUUID()}`
    setMessages(old => [...old, { message_id: userMessageId, session_id: sessionId ?? '', role: 'user', content, citations: [], created_at: new Date().toISOString() }])
    try {
      const queued = await api.ask(content, sessionId)
      setSessionId(queued.session_id)
      activeSessionRef.current = queued.session_id
      const shouldNavigate = !sessionId
      const assistantId = `pending-${queued.query_id}`
      const pendingAssistant: Message = { message_id: assistantId, session_id: queued.session_id, role: 'assistant', content: '', citations: [], created_at: new Date().toISOString() }
      pendingAssistantRef.current.set(queued.session_id, pendingAssistant)
      setMessages(old => [...old.map(item => item.message_id === userMessageId ? { ...item, session_id: queued.session_id } : item), pendingAssistant])
      let finished = false
      let stream: EventSource | undefined
      const close = () => { setLoading(false); setProgress(''); stream?.close() }
      const finalize = (result: QueryResult) => {
        if (finished) return
        finished = true
        pendingAssistantRef.current.delete(queued.session_id)
        setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: result.answer || result.error || '查询失败', citations: result.citations ?? [] } : item))
        setCitations(result.citations ?? []); close()
        if (shouldNavigate) navigate(`/?session=${queued.session_id}`, { replace: true })
        void onRefresh()
      }
      const handleError = (errorText: string) => {
        if (finished) return
        finished = true
        pendingAssistantRef.current.delete(queued.session_id)
        setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: errorText || '查询失败' } : item)); close()
      }
      const recover = async () => {
        for (let attempt = 0; attempt < 60 && !finished; attempt += 1) {
          try { const result = await api.result(queued.query_id); if (result.status !== 'processing') { finalize(result); return } } catch { /* 下一轮读取持久化状态 */ }
          await new Promise(resolve => window.setTimeout(resolve, 1000))
        }
        if (!finished) { finished = true; message.warning('查询仍在后台处理，可稍后从会话历史查看结果'); close() }
      }
      void recover()
      stream = new EventSource(api.eventsUrl(queued.query_id))
      stream.addEventListener('progress', event => { const { status } = JSON.parse((event as MessageEvent<string>).data); setProgress(status) })
      stream.addEventListener('delta', event => {
        const { delta } = JSON.parse((event as MessageEvent<string>).data)
        const pending = pendingAssistantRef.current.get(queued.session_id)
        if (pending) pendingAssistantRef.current.set(queued.session_id, { ...pending, content: pending.content + delta })
        setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: item.content + delta } : item))
      })
      stream.addEventListener('final', event => finalize(JSON.parse((event as MessageEvent<string>).data) as QueryResult))
      stream.addEventListener('error', event => { const payload = (event as MessageEvent<string>).data; if (!payload) return; try { handleError(JSON.parse(payload).error || '查询失败') } catch { handleError('查询失败') } })
      stream.onerror = () => { stream?.close() }
    } catch (error) {
      message.error(error instanceof Error ? error.message : '提交失败')
      if (sessionId) pendingAssistantRef.current.delete(sessionId)
      setMessages(old => old.filter(item => item.role !== 'assistant' || item.content)); setLoading(false); setProgress('')
    }
  }

  const openCitation = (citation: Citation) => {
    const document = documents.find(item => item.document_id === citation.document_id)
    if (document) setPreview({ document, versionId: citation.version_id, page: citation.locator.page ?? 1 })
  }
  const examples = ['华夏债券 C 的赎回费如何计算？', '中国货币政策执行报告的报告期是什么？']

  return <div className="workspace">
    <main className="chat-panel">
      <Breadcrumb className="page-breadcrumb" items={[{ title: '金融资料库' }, { title: '问答工作台' }]} />
      <div className="chat-intro">
        <div>
          <div className="eyebrow"><Badge status="processing" /> 资料驱动 · 可核对</div>
          <Title level={2}>金融资料问答</Title>
          <Paragraph className="intro-copy">从已导入的基金、理财、财报、政策和投教资料中查找答案，结论始终保留原文出处。</Paragraph>
        </div>
        <Tag className="offline-tag" icon={<SafetyCertificateOutlined />}>知识库内回答</Tag>
      </div>
      <div className="example-row" aria-label="问题示例">
        <Text type="secondary">试试：</Text>
        {examples.map(example => <Button key={example} size="small" type="default" onClick={() => setQuestion(example)}>{example}</Button>)}
      </div>
      <Card className="conversation-card" variant="borderless">
        {messages.length === 0 ? <div className="empty-conversation">
          <div className="empty-icon"><SearchOutlined /></div>
          <Title level={4}>从资料里找到答案</Title>
          <Text type="secondary">提问时尽量带上产品名称、报告名称或时间范围，回答会更准确。</Text>
        </div> : <div className="messages">
          {messages.map(item => <article className={`message ${item.role}`} key={item.message_id}>
            <div className="message-head">
              <Avatar size={30} className={item.role === 'user' ? 'avatar-user' : 'avatar-assistant'} icon={item.role === 'user' ? <MessageOutlined /> : <BookOutlined />} />
              <Text strong>{item.role === 'user' ? '你' : '掌柜智库'}</Text>
              <Text type="secondary" className="message-time">{formatTime(item.created_at)}</Text>
            </div>
            <div className={`message-body ${item.role === 'assistant' ? 'assistant-markdown' : ''}`}>{item.content ? (item.role === 'assistant' ? <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw, rehypeSanitize]}>{item.content}</ReactMarkdown> : item.content) : <Space><Spin size="small" /><Text type="secondary">正在整理回答…</Text></Space>}</div>
            {item.citations.length > 0 && <div className="message-citations">{item.citations.slice(0, 4).map((citation, index) => <Button key={citation.citation_id} size="small" type="default" icon={<FileSearchOutlined />} onClick={() => openCitation(citation)}>{index + 1} · {citation.title}{citation.locator.page ? ` · 第 ${citation.locator.page} 页` : ''}</Button>)}</div>}
          </article>)}
          {loading && <div className="thinking"><Spin size="small" /><Text type="secondary">{progress || '正在处理'}…</Text></div>}
        </div>}
      </Card>
      <Card className="composer-card" variant="borderless">
        <div className="composer-meta"><Text strong>向资料库提问</Text><Text type="secondary">Enter 发送 · Shift + Enter 换行</Text></div>
        <div className="composer-row">
          <Input.TextArea aria-label="输入金融资料问题" autoSize={{ minRows: 2, maxRows: 6 }} value={question} onChange={e => setQuestion(e.target.value)} onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); void ask() } }} placeholder="例如：这份基金资料中的管理费和托管费分别是多少？" />
          <Button className="send-button" type="primary" icon={<SendOutlined />} onClick={() => void ask()} loading={loading}>发送</Button>
        </div>
      </Card>
    </main>
    <aside className="sources-panel">
      <Card className="sources-card" variant="borderless" title={<div><div className="eyebrow">EVIDENCE RAIL</div><Title level={4}>回答依据</Title></div>} extra={citations.length ? <Badge count={citations.length} overflowCount={99} /> : null}>
        <div className="sources-scroll">
          {citations.length ? <List className="source-list" dataSource={citations} renderItem={(citation, index) => <List.Item className="source-item">
            <div className="source-title-row"><span className="source-index">{index + 1}</span><Text strong ellipsis={{ tooltip: citation.title }}>{citation.title}</Text></div>
            <Text type="secondary" className="source-location">{citation.locator.page ? `第 ${citation.locator.page} 页` : citation.locator.section || '资料正文'} · 版本 {citation.version_id.slice(0, 8)}</Text>
            <Paragraph className="source-excerpt" ellipsis={{ rows: 3 }}>{citation.locator.excerpt || '暂无摘录'}</Paragraph>
            <Button type="link" size="small" icon={<EyeOutlined />} onClick={() => openCitation(citation)}>打开原文</Button>
          </List.Item>} /> : <div className="sources-empty"><FileSearchOutlined /><Text type="secondary">完成一次提问后，这里会列出可核对的文件、版本和页码。</Text></div>}
        </div>
      </Card>
      <Card className="source-note" variant="borderless"><Space align="start"><SafetyCertificateOutlined className="note-icon" /><div><Text strong>回答边界</Text><Paragraph type="secondary">资料没有明确说明时会直接标注不足，不提供个性化买卖建议。</Paragraph></div></Space></Card>
    </aside>
    <Modal className="source-modal" open={Boolean(preview)} title={<Space><FileTextOutlined />{preview?.document.title}</Space>} onCancel={() => setPreview(undefined)} footer={null} width="min(980px, 92vw)">
      <div className="preview">{preview && <PdfPreview key={`${preview.versionId}:${preview.page}`} src={api.fileUrl(preview.document.document_id, preview.versionId)} initialPage={preview.page} />}</div>
    </Modal>
  </div>
}

function Documents({ documents, onRefresh }: { documents: Document[]; onRefresh: () => Promise<void> }) {
  const [uploading, setUploading] = useState(false)
  const [filters, setFilters] = useState<{ status?: string; document_type?: string }>({})
  const [tasks, setTasks] = useState<Record<string, ImportTask>>({})
  const [editing, setEditing] = useState<Document>()
  const [editTitle, setEditTitle] = useState('')
  const [editType, setEditType] = useState('unknown')
  const [editMetadata, setEditMetadata] = useState('{}')
  const visibleDocuments = documents.filter(doc => (!filters.status || doc.status === filters.status) && (!filters.document_type || doc.document_type === filters.document_type))
  const activeCount = documents.filter(item => item.status === 'active').length
  const processingCount = documents.filter(item => item.status === 'processing' || item.status === 'pending').length
  const failedCount = documents.filter(item => item.status === 'failed' || item.status === 'interrupted').length

  useEffect(() => {
    const ids = Object.keys(tasks).filter(id => !TERMINAL_TASK_STATES.has(tasks[id].status))
    if (!ids.length) return
    const timer = window.setInterval(() => {
      void Promise.all(ids.map(id => api.task(id).catch(() => undefined))).then(results => {
        let completed = false
        setTasks(old => {
          const next = { ...old }
          results.forEach(task => {
            if (!task) return
            if (!TERMINAL_TASK_STATES.has(old[task.task_id]?.status ?? '') && TERMINAL_TASK_STATES.has(task.status)) completed = true
            next[task.task_id] = task
          })
          return next
        })
        if (completed) void onRefresh()
      })
    }, 1500)
    return () => window.clearInterval(timer)
  }, [tasks, onRefresh])

  const trackTask = (task: ImportTask) => setTasks(old => ({ ...old, [task.task_id]: task }))
  const upload = async (file: File) => {
    setUploading(true)
    try { const { task } = await api.upload(file); trackTask(task); message.success('已创建导入任务'); await onRefresh() }
    catch (err) { message.error(err instanceof Error ? err.message : '上传失败') }
    finally { setUploading(false) }
    return false
  }
  const uploadVersion = async (document: Document, file: File) => {
    try { const { task } = await api.uploadVersion(document.document_id, file); trackTask(task); message.success(`已创建「${document.title}」的新版本任务`); await onRefresh() }
    catch (err) { message.error(err instanceof Error ? err.message : '版本上传失败') }
    return false
  }
  const openEdit = (document: Document) => { setEditing(document); setEditTitle(document.title); setEditType(document.document_type); setEditMetadata(JSON.stringify(document.metadata ?? {}, null, 2)) }
  const saveEdit = async () => {
    if (!editing || !editTitle.trim()) return
    try {
      const parsed = JSON.parse(editMetadata)
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error('元数据必须是 JSON 对象')
      await api.patchDocument(editing.document_id, { title: editTitle.trim(), document_type: editType, metadata: parsed as Record<string, unknown> })
      message.success('资料元数据已保存'); setEditing(undefined); await onRefresh()
    } catch (err) { message.error(err instanceof Error ? err.message : '保存失败，请检查元数据 JSON') }
  }
  const retry = async (task: ImportTask) => {
    try { await api.retryTask(task.task_id); setTasks(old => ({ ...old, [task.task_id]: { ...task, status: 'pending', stage: '等待重试' } })); message.success('已重新提交导入任务') }
    catch (err) { message.error(err instanceof Error ? err.message : '重试失败') }
  }

  return <div className="document-page">
    <Breadcrumb className="page-breadcrumb" items={[{ title: '金融资料库' }, { title: '资料管理' }]} />
    <div className="document-heading"><div><div className="eyebrow"><DatabaseOutlined /> KNOWLEDGE BASE</div><Title level={2}>金融资料库</Title><Paragraph className="intro-copy">统一管理原文件、解析状态和活动版本；停用的资料不会进入新的问答。</Paragraph></div><Upload beforeUpload={upload} showUploadList={false} accept=".pdf,.doc,.docx,.md"><Button type="primary" size="large" icon={<CloudUploadOutlined />} loading={uploading}>导入资料</Button></Upload></div>
    <div className="document-stats">
      <Card variant="borderless"><Statistic title="资料总数" value={documents.length} prefix={<FileTextOutlined />} /></Card>
      <Card variant="borderless"><Statistic title="已启用" value={activeCount} valueStyle={{ color: '#18794e' }} prefix={<CheckCircleOutlined />} /></Card>
      <Card variant="borderless"><Statistic title="处理中" value={processingCount} valueStyle={{ color: '#1d6fa5' }} prefix={<ClockCircleOutlined />} /></Card>
      <Card variant="borderless"><Statistic title="需处理" value={failedCount} valueStyle={{ color: failedCount ? '#b54708' : undefined }} prefix={<WarningOutlined />} /></Card>
    </div>
    <Card className="filter-card" variant="borderless">
      <div className="filter-heading"><Space><FilterOutlined /><Text strong>筛选资料</Text></Space><Text type="secondary">共 {visibleDocuments.length} 份</Text></div>
      <Space wrap>
        <Select allowClear placeholder="按状态筛选" style={{ width: 160 }} value={filters.status} onChange={value => setFilters(old => ({ ...old, status: value }))} options={[{ value: 'active', label: '已启用' }, { value: 'processing', label: '处理中' }, { value: 'failed', label: '处理失败' }, { value: 'disabled', label: '已停用' }]} />
        <Select allowClear placeholder="按类型筛选" style={{ width: 190 }} value={filters.document_type} onChange={value => setFilters(old => ({ ...old, document_type: value }))} options={DOCUMENT_TYPES} />
        {(filters.status || filters.document_type) && <Button type="link" onClick={() => setFilters({})}>清除筛选</Button>}
      </Space>
    </Card>
    <List className="documents-list" dataSource={visibleDocuments} locale={{ emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="还没有符合条件的金融资料" /> }} renderItem={doc => {
      const task = Object.values(tasks).find(item => item.document_id === doc.document_id && !TERMINAL_TASK_STATES.has(item.status))
      const failedTask = Object.values(tasks).find(item => item.document_id === doc.document_id && (item.status === 'failed' || item.status === 'interrupted'))
      const meta = statusMeta(doc.status)
      const publishDate = typeof doc.metadata.publish_date === 'string' ? doc.metadata.publish_date : undefined
      const reportPeriod = typeof doc.metadata.report_period === 'string' ? doc.metadata.report_period : undefined
      return <List.Item className="document-list-item"><Card className="document-card" variant="borderless" hoverable>
        <div className="document-card-main">
          <div className="document-icon"><FileTextOutlined /></div>
          <div className="document-info"><div className="document-title-row"><Title level={5} ellipsis={{ tooltip: doc.title }}>{doc.title}</Title><Tag icon={meta.icon} color={meta.color}>{meta.label}</Tag></div><Space wrap className="document-meta"><Tag bordered={false}>{documentTypeLabel(doc.document_type)}</Tag>{reportPeriod && <Text type="secondary">报告期：{reportPeriod}</Text>}{publishDate && <Text type="secondary">发布日期：{publishDate}</Text>}{task && <Text type="secondary">{task.stage}</Text>}</Space>{doc.error && <Alert className="document-error" type="error" showIcon message={doc.error} />}</div>
          <div className="document-actions"><Space wrap>
            <Button type="default" icon={<EditOutlined />} onClick={() => openEdit(doc)}>修正</Button>
            <Upload beforeUpload={file => uploadVersion(doc, file)} showUploadList={false} accept=".pdf,.doc,.docx,.md"><Button icon={<ReloadOutlined />}>新版本</Button></Upload>
            <Button icon={<EyeOutlined />} href={api.fileUrl(doc.document_id, doc.active_version_id ?? undefined)} target="_blank">原文</Button>
            {doc.status !== 'disabled' && <Button danger type="text" onClick={() => api.patchDocument(doc.document_id, { status: 'disabled' }).then(onRefresh).catch(err => message.error(err instanceof Error ? err.message : '停用失败'))}>停用</Button>}
            {failedTask && <Button type="link" icon={<ReloadOutlined />} onClick={() => void retry(failedTask)}>重试处理</Button>}
          </Space></div>
        </div>
        <Divider className="document-divider" />
        <div className="version-row"><Text type="secondary">版本记录</Text><Space wrap>{doc.versions.map(version => <Tag key={version.version_id} icon={version.version_id === doc.active_version_id ? <CheckCircleOutlined /> : undefined} color={version.version_id === doc.active_version_id ? 'green' : undefined}>{version.original_name}{version.version_id === doc.active_version_id ? ' · 当前活动版本' : ''}</Tag>)}</Space></div>
      </Card></List.Item>
    }} />
    <Modal open={Boolean(editing)} title={<Space><EditOutlined />修正资料元数据</Space>} onCancel={() => setEditing(undefined)} onOk={() => void saveEdit()} okText="保存修改" cancelText="取消">
      <Space direction="vertical" size={14} style={{ width: '100%' }}><Input addonBefore="标题" value={editTitle} onChange={event => setEditTitle(event.target.value)} placeholder="资料标题" /><Select value={editType} onChange={setEditType} options={DOCUMENT_TYPES} style={{ width: '100%' }} /><div><Text strong>元数据 JSON</Text><Input.TextArea className="metadata-editor" value={editMetadata} onChange={event => setEditMetadata(event.target.value)} autoSize={{ minRows: 8, maxRows: 16 }} /></div></Space>
    </Modal>
  </div>
}

function Root() {
  return <ConfigProvider theme={{
    token: { colorPrimary: '#0b7285', colorInfo: '#0b7285', colorSuccess: '#18794e', colorWarning: '#b54708', colorError: '#c2413d', colorText: '#17324d', colorTextSecondary: '#63798f', colorBgLayout: '#eef3f6', colorBgContainer: '#ffffff', borderRadius: 10, fontFamily: "'Noto Sans SC', 'Microsoft YaHei', sans-serif", controlHeight: 40 },
    components: { Layout: { headerBg: '#ffffff', siderBg: '#ffffff' }, Menu: { itemSelectedBg: '#e7f5f4', itemSelectedColor: '#075e63', itemBorderRadius: 9 }, Card: { paddingLG: 22 }, Button: { fontWeight: 600 }, Input: { activeBorderColor: '#0b7285', hoverBorderColor: '#5aaeb4' } },
  }}><AntApp><BrowserRouter><AppShell /></BrowserRouter></AntApp></ConfigProvider>
}

createRoot(document.getElementById('root')!).render(<Root />)
