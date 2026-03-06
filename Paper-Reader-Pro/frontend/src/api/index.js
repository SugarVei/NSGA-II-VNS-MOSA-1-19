/**
 * API 接口封装
 * 与后端 /api/upload_and_parse 和 /api/chat 两个核心接口对接
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 上传+解析+翻译可能较慢，设置 5 分钟超时
})

// ===== 核心接口 1：上传并解析翻译 =====

/**
 * 上传 PDF → 后端自动解析坐标 + 调用 OpenAI 翻译 → 返回双语 JSON
 * @param {File} file - PDF 文件对象
 * @returns 包含所有页面、坐标、原文和译文的完整 JSON
 */
export async function uploadAndParse(file) {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/upload_and_parse', formData)
  return data
}

// ===== 核心接口 2：AI 对话 (SSE 流式) =====

/**
 * 向 /api/chat 发送问题，通过 SSE 流式接收 Gemini 的回答
 * @param {Object} params - { filename, question, selected_text?, chat_history? }
 * @param {Function} onChunk - 每收到一个文本块时的回调 (text) => void
 * @param {Function} onDone - AI 回答完成时的回调 () => void
 * @param {Function} onError - 出错时的回调 (errorMessage) => void
 */
export async function analyzeChat(params, onChunk, onDone, onError) {
  try {
    const response = await fetch('/api/chat', {
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
          try {
            const parsed = JSON.parse(line.slice(6))
            if (parsed.done) {
              onDone?.()
            } else if (parsed.error) {
              onError?.(parsed.error)
            } else if (parsed.text) {
              onChunk?.(parsed.text)
            }
          } catch { /* 忽略 JSON 解析错误 */ }
        }
      }
    }
  } catch (err) {
    onError?.(err.message)
  }
}

// ===== 辅助接口 =====

/** 获取已上传的文献列表 */
export async function listPapers() {
  const { data } = await api.get('/papers/list')
  return data.papers
}

/** 删除文献 */
export async function deletePaper(filename) {
  await api.delete(`/papers/${filename}`)
}

/** 保存 API Key 配置到后端运行时 */
export async function saveConfig(config) {
  const { data } = await api.post('/config', config)
  return data
}

/** 查询 API Key 配置状态 */
export async function getConfig() {
  const { data } = await api.get('/config')
  return data
}
