import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { Alert, App as AntApp, Button, ConfigProvider, Empty, Input, Layout, List, Menu, Modal, Select, Space, Spin, Tag, Typography, Upload, message } from 'antd'
import { BookOutlined, FileTextOutlined, FolderOpenOutlined, LeftOutlined, PlusOutlined, RightOutlined, SendOutlined, UploadOutlined } from '@ant-design/icons'
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy } from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { api } from './api'
import type { Citation, Document, ImportTask, Message, QueryResult, Session } from './types'
import './styles.css'

GlobalWorkerOptions.workerSrc = pdfWorkerUrl

const { Sider, Content } = Layout
const { Text, Title, Paragraph } = Typography

type PreviewState = { document: Document; versionId: string; page: number }

function AppShell() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const { pathname, search } = useLocation()
  const navigate = useNavigate()
  const refresh = async () => {
    const [sessionResult, documentResult] = await Promise.all([api.sessions(), api.documents()])
    setSessions(sessionResult.items)
    setDocuments(documentResult.items)
  }
  useEffect(() => { void refresh() }, [])
  return <Layout className="frame">
    <Sider width={272} theme="light" className="side">
      <div className="brand"><BookOutlined /><span>掌柜智库</span><small>金融资料查询</small></div>
      <Link to="/"><Button type="primary" icon={<PlusOutlined />} block>新建对话</Button></Link>
      <Menu mode="inline" selectedKeys={pathname.startsWith('/documents') ? ['documents'] : ['chat']} items={[
        { key: 'chat', icon: <SendOutlined />, label: <Link to="/">问答工作台</Link> },
        { key: 'documents', icon: <FolderOpenOutlined />, label: <Link to="/documents">资料管理</Link> },
      ]} />
      <div className="side-label">最近会话</div>
      <List size="small" dataSource={sessions.slice(0, 8)} locale={{ emptyText: '暂无会话' }} renderItem={item => <List.Item className="session-item" actions={[
        <Button key="delete" type="text" size="small" danger onClick={() => api.deleteSession(item.session_id).then(async () => {
          if (new URLSearchParams(search).get('session') === item.session_id) navigate('/')
          await refresh()
        }).catch(err => message.error(err instanceof Error ? err.message : '删除失败'))}>删除</Button>,
      ]}><Link to={`/?session=${item.session_id}`}>{item.title}</Link></List.Item>} />
    </Sider>
    <Content><Routes>
      <Route path="/" element={<Chat documents={documents} onRefresh={refresh} />} />
      <Route path="/documents" element={<Documents documents={documents} onRefresh={refresh} />} />
    </Routes></Content>
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
    setPdf(undefined)
    setPageCount(0)
    setLoading(true)
    setError('')
    void fetch(src, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error(`原文加载失败（HTTP ${response.status}）`)
        return response.arrayBuffer()
      })
      .then(data => getDocument({ data }).promise)
      .then(result => {
        loadedPdf = result
        if (cancelled) {
          void result.destroy()
          return
        }
        setPdf(result)
        setPageCount(result.numPages)
        setPage(current => Math.min(Math.max(1, initialPage), result.numPages))
      })
      .catch(err => {
        if (!cancelled && err?.name !== 'AbortError') setError(err instanceof Error ? err.message : 'PDF 原文加载失败')
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => {
      cancelled = true
      controller.abort()
      if (loadedPdf) void loadedPdf.destroy()
    }
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
      canvas.width = viewport.width
      canvas.height = viewport.height
      canvas.style.aspectRatio = `${viewport.width} / ${viewport.height}`
      return pdfPage.render({ canvasContext: context, viewport }).promise
    }).catch(err => {
      if (!cancelled) setError(err instanceof Error ? err.message : 'PDF 页面渲染失败')
    }).finally(() => { if (!cancelled) setRendering(false) })
    return () => { cancelled = true }
  }, [pdf, page])

  if (loading) return <div className="pdf-state"><Spin /> <Text type="secondary">正在加载原文…</Text></div>
  if (error) return <Alert type="error" showIcon message="原文预览失败" description={error} />
  if (!pdf || !pageCount) return <Empty description="没有可预览的 PDF 页面" />
  return <div className="pdf-viewer">
    <div className="pdf-toolbar">
      <Space>
        <Button size="small" icon={<LeftOutlined />} disabled={page <= 1 || rendering} onClick={() => setPage(value => Math.max(1, value - 1))}>上一页</Button>
        <Text>第 {page} / {pageCount} 页</Text>
        <Button size="small" icon={<RightOutlined />} disabled={page >= pageCount || rendering} onClick={() => setPage(value => Math.min(pageCount, value + 1))}>下一页</Button>
      </Space>
      {rendering && <Text type="secondary">正在渲染…</Text>}
    </div>
    <div className="pdf-canvas-wrap"><canvas ref={canvasRef} aria-label={`PDF 第 ${page} 页`} /></div>
  </div>
}

function Chat({ documents, onRefresh }: { documents: Document[]; onRefresh: () => Promise<void> }) {
  const [sessionId, setSessionId] = useState<string>()
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [preview, setPreview] = useState<PreviewState>()
  const navigate = useNavigate()
  const { search } = useLocation()
  const selectedSessionId = new URLSearchParams(search).get('session')

  useEffect(() => {
    let cancelled = false
    if (!selectedSessionId) {
      setSessionId(undefined)
      setMessages([])
      setCitations([])
      return () => { cancelled = true }
    }
    setSessionId(selectedSessionId)
    setCitations([])
    void api.messages(selectedSessionId)
      .then(result => { if (!cancelled) setMessages(result.items) })
      .catch(() => { if (!cancelled) setMessages([]) })
    return () => { cancelled = true }
  }, [selectedSessionId])

  const ask = async () => {
    const content = question.trim()
    if (!content || loading) return
    setQuestion('')
    setLoading(true)
    setCitations([])
    setProgress('理解问题')
    setMessages(old => [...old, { message_id: crypto.randomUUID(), session_id: sessionId ?? '', role: 'user', content, citations: [], created_at: new Date().toISOString() }])
    try {
      const queued = await api.ask(content, sessionId)
      setSessionId(queued.session_id)
      const shouldNavigate = !sessionId
      const assistantId = crypto.randomUUID()
      setMessages(old => [...old, { message_id: assistantId, session_id: queued.session_id, role: 'assistant', content: '', citations: [], created_at: new Date().toISOString() }])
      let finished = false
      let stream: EventSource | undefined
      const close = () => {
        setLoading(false)
        setProgress('')
        stream?.close()
      }
      const finalize = (result: QueryResult) => {
        if (finished) return
        finished = true
        setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: result.answer || result.error || '查询失败', citations: result.citations ?? [] } : item))
        setCitations(result.citations ?? [])
        close()
        if (shouldNavigate) navigate(`/?session=${queued.session_id}`, { replace: true })
        void onRefresh()
      }
      const handleError = (errorText: string) => {
        if (finished) return
        finished = true
        setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: errorText || '查询失败' } : item))
        close()
      }
      const recover = async () => {
        for (let attempt = 0; attempt < 60 && !finished; attempt += 1) {
          try {
            const result = await api.result(queued.query_id)
            if (result.status !== 'processing') { finalize(result); return }
          } catch { /* 网络暂时不可用，下一轮继续读取持久化状态 */ }
          await new Promise(resolve => window.setTimeout(resolve, 1000))
        }
        if (!finished) {
          finished = true
          message.warning('查询仍在后台处理，可稍后从会话历史查看结果')
          close()
        }
      }
      // SSE 是实时通道，同时启动持久化状态轮询，避免首次连接或浏览器切页时丢失 final 事件。
      void recover()
      stream = new EventSource(api.eventsUrl(queued.query_id))
      stream.addEventListener('progress', event => { const { status } = JSON.parse((event as MessageEvent<string>).data); setProgress(status) })
      stream.addEventListener('delta', event => { const { delta } = JSON.parse((event as MessageEvent<string>).data); setMessages(old => old.map(item => item.message_id === assistantId ? { ...item, content: item.content + delta } : item)) })
      stream.addEventListener('final', event => finalize(JSON.parse((event as MessageEvent<string>).data) as QueryResult))
      stream.addEventListener('error', event => {
        const payload = (event as MessageEvent<string>).data
        if (!payload) return
        try { handleError(JSON.parse(payload).error || '查询失败') } catch { handleError('查询失败') }
      })
      stream.onerror = () => { stream?.close() }
    } catch (error) {
      message.error(error instanceof Error ? error.message : '提交失败')
      setMessages(old => old.filter(item => item.role !== 'assistant' || item.content))
      setLoading(false)
      setProgress('')
    }
  }

  const openCitation = (citation: Citation) => {
    const document = documents.find(item => item.document_id === citation.document_id)
    if (document) setPreview({ document, versionId: citation.version_id, page: citation.locator.page ?? 1 })
  }
  return <div className="workspace">
    <main className="chat">
      <header><Text className="eyebrow">知识库内资料 · 可核对引用</Text><Title level={2}>金融资料问答</Title><Paragraph>可查询基金产品、理财风险揭示、公司财报、政策和投资者教育资料。</Paragraph></header>
      <div className="messages">
        {messages.length === 0 ? <Empty description="从一条具体的问题开始，例如：华夏债券 C 的赎回费如何计算？" /> : messages.map(item => <article className={`bubble ${item.role}`} key={item.message_id}>
          <Text type="secondary">{item.role === 'user' ? '你' : '掌柜智库'}</Text>
          <Paragraph>{item.content}</Paragraph>
          {item.citations.length > 0 && <Space wrap>{item.citations.map(citation => <Tag key={citation.citation_id} color="cyan" onClick={() => openCitation(citation)}>{citation.title}{citation.locator.page ? ` · 第 ${citation.locator.page} 页` : ''}</Tag>)}</Space>}
        </article>)}
        {loading && <div className="thinking"><Spin size="small" /> {progress || '正在处理'}…</div>}
      </div>
      <div className="composer"><Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} value={question} onChange={e => setQuestion(e.target.value)} onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); void ask() } }} placeholder="输入金融资料问题；系统只依据已导入资料回答" /><Button type="primary" icon={<SendOutlined />} onClick={() => void ask()} loading={loading}>发送</Button></div>
    </main>
    <aside className="sources"><Text className="eyebrow">来源预览</Text><Title level={4}>回答依据</Title>{citations.length ? <List dataSource={citations} renderItem={citation => <List.Item><div><Text strong>{citation.title}</Text><br /><Text type="secondary">{citation.locator.page ? `第 ${citation.locator.page} 页` : citation.locator.section || '资料正文'}</Text><Paragraph ellipsis={{ rows: 3 }}>{citation.locator.excerpt}</Paragraph></div></List.Item>} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="回答后在此查看引用" />}</aside>
    <Modal open={Boolean(preview)} title={preview?.document.title} onCancel={() => setPreview(undefined)} footer={null} width="80vw"><div className="preview">{preview && <PdfPreview key={`${preview.versionId}:${preview.page}`} src={api.fileUrl(preview.document.document_id, preview.versionId)} initialPage={preview.page} />}</div></Modal>
  </div>
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

function Documents({ documents, onRefresh }: { documents: Document[]; onRefresh: () => Promise<void> }) {
  const [uploading, setUploading] = useState(false)
  const [filters, setFilters] = useState<{ status?: string; document_type?: string }>({})
  const [tasks, setTasks] = useState<Record<string, ImportTask>>({})
  const [editing, setEditing] = useState<Document>()
  const [editTitle, setEditTitle] = useState('')
  const [editType, setEditType] = useState('unknown')
  const [editMetadata, setEditMetadata] = useState('{}')
  const visibleDocuments = documents.filter(doc => (!filters.status || doc.status === filters.status) && (!filters.document_type || doc.document_type === filters.document_type))

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
    try {
      const { task } = await api.upload(file)
      trackTask(task)
      message.success(`已创建导入任务 ${task.task_id.slice(0, 8)}`)
      await onRefresh()
    } catch (err) { message.error(err instanceof Error ? err.message : '上传失败') }
    finally { setUploading(false) }
    return false
  }
  const uploadVersion = async (document: Document, file: File) => {
    try {
      const { task } = await api.uploadVersion(document.document_id, file)
      trackTask(task)
      message.success(`已创建 ${document.title} 的新版本任务`)
      await onRefresh()
    } catch (err) { message.error(err instanceof Error ? err.message : '版本上传失败') }
    return false
  }
  const openEdit = (document: Document) => {
    setEditing(document)
    setEditTitle(document.title)
    setEditType(document.document_type)
    setEditMetadata(JSON.stringify(document.metadata ?? {}, null, 2))
  }
  const saveEdit = async () => {
    if (!editing || !editTitle.trim()) return
    try {
      const parsed = JSON.parse(editMetadata)
      if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') throw new Error('元数据必须是 JSON 对象')
      await api.patchDocument(editing.document_id, { title: editTitle.trim(), document_type: editType, metadata: parsed as Record<string, unknown> })
      message.success('资料元数据已保存')
      setEditing(undefined)
      await onRefresh()
    } catch (err) { message.error(err instanceof Error ? err.message : '保存失败，请检查元数据 JSON') }
  }
  const retry = async (task: ImportTask) => {
    try {
      await api.retryTask(task.task_id)
      setTasks(old => ({ ...old, [task.task_id]: { ...task, status: 'pending', stage: '等待重试' } }))
      message.success('已重新提交导入任务')
    } catch (err) { message.error(err instanceof Error ? err.message : '重试失败') }
  }
  return <div className="document-page">
    <header><Text className="eyebrow">资料管理</Text><Title level={2}>金融资料库</Title><Paragraph>每份资料独立记录版本、解析状态与来源位置。停用后不会再参与新的问答。</Paragraph></header>
    <Space wrap>
      <Upload beforeUpload={upload} showUploadList={false} accept=".pdf,.doc,.docx,.md"><Button type="primary" icon={<UploadOutlined />} loading={uploading}>导入资料</Button></Upload>
      <Select allowClear placeholder="按状态筛选" style={{ width: 150 }} value={filters.status} onChange={value => setFilters(old => ({ ...old, status: value }))} options={[{ value: 'active', label: '已启用' }, { value: 'processing', label: '处理中' }, { value: 'failed', label: '失败' }, { value: 'disabled', label: '已停用' }]} />
      <Select allowClear placeholder="按类型筛选" style={{ width: 180 }} value={filters.document_type} onChange={value => setFilters(old => ({ ...old, document_type: value }))} options={DOCUMENT_TYPES} />
    </Space>
    <List className="document-list" dataSource={visibleDocuments} locale={{ emptyText: '还没有符合条件的金融资料' }} renderItem={doc => {
      const task = Object.values(tasks).find(item => item.document_id === doc.document_id && !TERMINAL_TASK_STATES.has(item.status))
      const failedTask = Object.values(tasks).find(item => item.document_id === doc.document_id && (item.status === 'failed' || item.status === 'interrupted'))
      const publishDate = typeof doc.metadata.publish_date === 'string' ? doc.metadata.publish_date : undefined
      return <List.Item actions={[
        <Button key="edit" type="link" onClick={() => openEdit(doc)}>修正元数据</Button>,
        <Upload key="version" beforeUpload={file => uploadVersion(doc, file)} showUploadList={false} accept=".pdf,.doc,.docx,.md"><Button type="link">上传新版本</Button></Upload>,
        <a key="file" href={api.fileUrl(doc.document_id, doc.active_version_id ?? undefined)} target="_blank" rel="noreferrer">查看原文</a>,
        doc.status !== 'disabled' && <Button key="disable" type="link" danger onClick={() => api.disable(doc.document_id).then(onRefresh).catch(err => message.error(err instanceof Error ? err.message : '停用失败'))}>停用</Button>,
        failedTask && <Button key="retry" type="link" onClick={() => void retry(failedTask)}>重试</Button>,
      ]}>
        <List.Item.Meta avatar={<FileTextOutlined className="doc-icon" />} title={doc.title} description={<Space wrap><Tag>{DOCUMENT_TYPES.find(item => item.value === doc.document_type)?.label ?? doc.document_type}</Tag><Tag color={doc.status === 'active' ? 'green' : doc.status === 'failed' ? 'red' : 'gold'}>{doc.status}</Tag>{publishDate && <Text type="secondary">发布日期：{publishDate}</Text>}{task && <Text type="secondary">{task.stage}</Text>}</Space>} />
        <Space direction="vertical" align="end"><Space wrap>{doc.versions.map(version => <Tag key={version.version_id} color={version.version_id === doc.active_version_id ? 'green' : undefined}>{version.original_name}{version.version_id === doc.active_version_id ? ' · 当前' : ''}</Tag>)}</Space>{doc.error && <Text type="danger">{doc.error}</Text>}</Space>
      </List.Item>
    }} />
    <Modal open={Boolean(editing)} title="修正资料元数据" onCancel={() => setEditing(undefined)} onOk={() => void saveEdit()} okText="保存">
      <Space direction="vertical" style={{ width: '100%' }}>
        <Input value={editTitle} onChange={event => setEditTitle(event.target.value)} placeholder="资料标题" />
        <Select value={editType} onChange={setEditType} options={DOCUMENT_TYPES} style={{ width: '100%' }} />
        <Input.TextArea value={editMetadata} onChange={event => setEditMetadata(event.target.value)} autoSize={{ minRows: 8, maxRows: 16 }} placeholder="JSON 元数据" />
      </Space>
    </Modal>
  </div>
}

function Root() { return <ConfigProvider theme={{ token: { colorPrimary: '#087e8b', borderRadius: 8, colorText: '#102a43' } }}><AntApp><BrowserRouter><AppShell /></BrowserRouter></AntApp></ConfigProvider> }

createRoot(document.getElementById('root')!).render(<Root />)