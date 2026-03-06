/**
 * API 接口封装
 * 统一管理所有后端请求
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 翻译请求可能较慢，设置 2 分钟超时
})

// ===== 文献管理 =====

/** 上传 PDF 文件 */
export async function uploadPaper(file) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/papers/upload', formData)
  return data
}

/** 获取文献列表 */
export async function listPapers() {
  const { data } = await api.get('/papers/list')
  return data.papers
}

/** 解析 PDF 文件（获取坐标数据） */
export async function parsePaper(filename) {
  const { data } = await api.get(`/papers/parse/${filename}`)
  return data
}

/** 删除文献 */
export async function deletePaper(filename) {
  await api.delete(`/papers/${filename}`)
}

// ===== 翻译 =====

/** 翻译指定页面 */
export async function translatePage(filename, pageNumber) {
  const { data } = await api.get(`/translate/${filename}/page/${pageNumber}`)
  return data
}

// ===== 分析（SSE 流式） =====

/**
 * 发送分析请求并以 SSE 流式接收回答
 * @param {Object} params - { filename, question, selected_text?, chat_history? }
 * @param {Function} onChunk - 每收到一个文本块时的回调 (text) => void
 * @param {Function} onDone - 完成时的回调 () => void
 * @param {Function} onError - 错误时的回调 (error) => void
 */
export async function analyzeChat(params, onChunk, onDone, onError) {
  try {
    const response = await fetch('/api/analyze/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    })

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() // 保留不完整的行

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6)
          try {
            const parsed = JSON.parse(jsonStr)
            if (parsed.done) {
              onDone?.()
            } else if (parsed.error) {
              onError?.(parsed.error)
            } else if (parsed.text) {
              onChunk?.(parsed.text)
            }
          } catch { /* 忽略解析错误 */ }
        }
      }
    }
  } catch (err) {
    onError?.(err.message)
  }
}

// ===== 配置 =====

/** 保存 API Key 配置到后端 */
export async function saveConfig(config) {
  const { data } = await api.post('/analyze/config', config)
  return data
}

/** 获取当前配置状态 */
export async function getConfig() {
  const { data } = await api.get('/analyze/config')
  return data
}
