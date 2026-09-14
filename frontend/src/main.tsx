import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { App as AntApp, Button, ConfigProvider, Empty, Input, Layout, List, Menu, Modal, Space, Spin, Tag, Typography, Upload, message } from 'antd'
import { BookOutlined, FileTextOutlined, FolderOpenOutlined, PlusOutlined, SendOutlined, UploadOutlined } from '@ant-design/icons'
import { api } from './api'
import type { Citation, Document, Message, QueryResult, Session } from './types'
import './styles.css'

const { Sider, Content } = Layout
const { Text, Title, Paragraph } = Typography

function AppShell() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const { pathname } = useLocation()
  const refresh = async () => { setSessions((await api.sessions()).items); setDocuments((await api.documents()).items) }
  useEffect(() => { void refresh() }, [])
  return <Layout className="frame"><Sider width={272} theme="light" className="side"><div className="brand"><BookOutlined /><span>掌柜智库</span><small>金融资料查询</small></div><Link to="/"><Button type="primary" icon={<PlusOutlined />} block>新建对话</Button></Link><Menu mode="inline" selectedKeys={pathname.startsWith('/documents') ? ['documents'] : ['chat']} items={[{ key: 'chat', icon: <SendOutlined />, label: <Link to="/">问答工作台</Link> }, { key: 'documents', icon: <FolderOpenOutlined />, label: <Link to="/documents">资料管理</Link> }]} /><div className="side-label">最近会话</div><List size="small" dataSource={sessions.slice(0, 8)} locale={{ emptyText: '暂无会话' }} renderItem={item => <List.Item className="session-item"><Link to={`/?session=${item.session_id}`}>{item.title}</Link></List.Item>} /></Sider><Content><Routes><Route path="/" element={<Chat documents={documents} />} /><Route path="/documents" element={<Documents documents={documents} onRefresh={refresh} />} /></Routes></Content></Layout>
}

function Chat({ documents }: { documents: Document[] }) {
  const [sessionId, setSessionId] = useState<string>()
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [citations, setCitations] = useState<Citation[]>([])
  const [preview, setPreview] = useState<Document>()
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
    const content = question.trim(); if (!content || loading) return
    setQuestion(''); setLoading(true); setMessages(old => [...old, { message_id: crypto.randomUUID(), session_id: sessionId ?? '', role: 'user', content, citations: [], created_at: new Date().toISOString() }])
    try {
      const queued = await api.ask(content, sessionId); setSessionId(queued.session_id); navigate(`/?session=${queued.session_id}`, { replace: true })
      const stream = new EventSource(api.eventsUrl(queued.query_id))
      stream.addEventListener('final', event => { const result = JSON.parse((event as MessageEvent<string>).data) as QueryResult; setCitations(result.citations ?? []); setMessages(old => [...old, { message_id: crypto.randomUUID(), session_id: result.session_id, role: 'assistant', content: result.answer || result.error || '查询失败', citations: result.citations ?? [], created_at: new Date().toISOString() }]); setLoading(false); stream.close() })
      stream.onerror = () => { stream.close(); api.result(queued.query_id).then(result => { if (result.status !== 'processing') { setMessages(old => [...old, { message_id: crypto.randomUUID(), session_id: result.session_id, role: 'assistant', content: result.answer || result.error || '查询失败', citations: result.citations ?? [], created_at: new Date().toISOString() }]); setLoading(false) } }) }
    } catch (error) { message.error(error instanceof Error ? error.message : '提交失败'); setLoading(false) }
  }
  return <div className="workspace"><main className="chat"><header><Text className="eyebrow">知识库内资料 · 可核对引用</Text><Title level={2}>金融资料问答</Title><Paragraph>可查询基金产品、理财风险揭示、公司财报、政策和投资者教育资料。</Paragraph></header><div className="messages">{messages.length === 0 ? <Empty description="从一条具体的问题开始，例如：华夏债券 C 的赎回费如何计算？" /> : messages.map(item => <article className={`bubble ${item.role}`} key={item.message_id}><Text type="secondary">{item.role === 'user' ? '你' : '掌柜智库'}</Text><Paragraph>{item.content}</Paragraph>{item.citations.length > 0 && <Space wrap>{item.citations.map(c => <Tag key={c.citation_id} color="cyan" onClick={() => setPreview(documents.find(d => d.document_id === c.document_id))}>{c.title}{c.locator.page ? ` · 第 ${c.locator.page} 页` : ''}</Tag>)}</Space>}</article>)}{loading && <div className="thinking"><Spin size="small" /> 正在查找资料并整理回答…</div>}</div><div className="composer"><Input.TextArea autoSize={{ minRows: 2, maxRows: 6 }} value={question} onChange={e => setQuestion(e.target.value)} onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); void ask() } }} placeholder="输入金融资料问题；系统只依据已导入资料回答" /><Button type="primary" icon={<SendOutlined />} onClick={() => void ask()} loading={loading}>发送</Button></div></main><aside className="sources"><Text className="eyebrow">来源预览</Text><Title level={4}>回答依据</Title>{citations.length ? <List dataSource={citations} renderItem={c => <List.Item><div><Text strong>{c.title}</Text><br /><Text type="secondary">{c.locator.page ? `第 ${c.locator.page} 页` : c.locator.section || '资料正文'}</Text><Paragraph ellipsis={{ rows: 3 }}>{c.locator.excerpt}</Paragraph></div></List.Item>} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="回答后在此查看引用" />}</aside><Modal open={Boolean(preview)} title={preview?.title} onCancel={() => setPreview(undefined)} footer={null} width="80vw"><iframe className="preview" src={preview ? api.fileUrl(preview.document_id) : ''} title="资料原文" /></Modal></div>
}

function Documents({ documents, onRefresh }: { documents: Document[]; onRefresh: () => Promise<void> }) {
  const [uploading, setUploading] = useState(false)
  const upload = async (file: File) => { setUploading(true); try { const { task } = await api.upload(file); message.success(`已创建导入任务 ${task.task_id.slice(0, 8)}`); await onRefresh() } catch (err) { message.error(err instanceof Error ? err.message : '上传失败') } finally { setUploading(false) }; return false }
  return <div className="document-page"><header><Text className="eyebrow">资料管理</Text><Title level={2}>金融资料库</Title><Paragraph>每份资料独立记录版本、解析状态与来源位置。停用后不会再参与新的问答。</Paragraph></header><Upload beforeUpload={upload} showUploadList={false} accept=".pdf,.doc,.docx,.md"><Button type="primary" icon={<UploadOutlined />} loading={uploading}>导入资料</Button></Upload><List className="document-list" dataSource={documents} locale={{ emptyText: '还没有导入金融资料' }} renderItem={doc => <List.Item actions={[<a key="file" href={api.fileUrl(doc.document_id)} target="_blank">查看原文</a>, doc.status !== 'disabled' && <a key="disable" onClick={() => api.disable(doc.document_id).then(onRefresh)}>停用</a>]}><List.Item.Meta avatar={<FileTextOutlined className="doc-icon" />} title={doc.title} description={<Space wrap><Tag>{doc.document_type}</Tag><Tag color={doc.status === 'active' ? 'green' : doc.status === 'failed' ? 'red' : 'gold'}>{doc.status}</Tag>{doc.metadata.publish_date && <Text type="secondary">{doc.metadata.publish_date}</Text>}</Space>} />{doc.error && <Text type="danger">{doc.error}</Text>}</List.Item>} /></div>
}

function Root() { return <ConfigProvider theme={{ token: { colorPrimary: '#087e8b', borderRadius: 8, colorText: '#102a43' } }}><AntApp><BrowserRouter><AppShell /></BrowserRouter></AntApp></ConfigProvider> }

createRoot(document.getElementById('root')!).render(<Root />)
