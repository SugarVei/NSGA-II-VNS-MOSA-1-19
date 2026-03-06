<template>
  <!--
    分析模式 — 左文右聊沉浸式布局
    ===============================
    左侧：PDF 阅读视图（与阅读模式共用 PDF.js 渲染逻辑）
    右侧：Claude 风格对话窗口，底部输入框，消息区分 User/AI 气泡

    SSE 流式输出：
    - 请求 /api/chat 时接收 Server-Sent Events
    - 每个 data chunk 实时追加到 streamBuffer → 打字机效果
    - 完成后将完整内容移入 messages 数组
  -->
  <div class="h-full flex gap-4">

    <!-- ========== 左侧：PDF 阅读视图 ========== -->
    <div class="w-1/2 overflow-y-auto rounded-2xl bg-white shadow-sm">
      <!-- 分页控制 -->
      <div class="sticky top-0 z-10 flex items-center justify-center gap-4 py-3 bg-white/90 backdrop-blur-sm border-b border-gray-100">
        <button
          @click="goToPage(currentPage - 1)"
          :disabled="currentPage <= 1"
          class="px-3 py-1 text-sm rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
        >&larr;</button>
        <span class="text-sm text-gray-400 tabular-nums">{{ currentPage }} / {{ totalPages }}</span>
        <button
          @click="goToPage(currentPage + 1)"
          :disabled="currentPage >= totalPages"
          class="px-3 py-1 text-sm rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
        >&rarr;</button>
      </div>
      <div class="p-4">
        <canvas ref="pdfCanvas" class="pdf-page-canvas w-full"></canvas>
      </div>
    </div>

    <!-- ========== 右侧：Claude 风格对话窗口 ========== -->
    <div class="w-1/2 rounded-2xl bg-white shadow-sm flex flex-col">

      <!-- 头部 -->
      <div class="px-5 py-3.5 border-b border-gray-100 flex-shrink-0">
        <h3 class="text-sm font-semibold text-gray-800">AI 论文分析</h3>
        <p class="text-xs text-gray-400 mt-0.5">基于论文全文上下文的智能问答</p>
      </div>

      <!-- 消息区域 -->
      <div ref="chatContainer" class="flex-1 overflow-y-auto px-5 py-4 space-y-5">

        <!-- 空状态：推荐问题 -->
        <div v-if="messages.length === 0 && !streaming" class="flex items-center justify-center h-full">
          <div class="text-center space-y-5 max-w-sm">
            <div class="w-14 h-14 mx-auto rounded-2xl bg-[#F9F9F9] flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-7 h-7 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"/>
              </svg>
            </div>
            <p class="text-sm text-gray-400">向 AI 提问关于这篇论文的问题</p>
            <div class="space-y-2">
              <button
                v-for="q in suggestedQuestions"
                :key="q"
                @click="sendMessage(q)"
                class="block w-full text-left px-4 py-2.5 text-sm text-gray-600 bg-[#F9F9F9] rounded-xl hover:bg-gray-100 transition-colors"
              >{{ q }}</button>
            </div>
          </div>
        </div>

        <!-- 消息气泡 -->
        <template v-for="(msg, idx) in messages" :key="idx">
          <!-- 用户消息 — 右对齐 -->
          <div v-if="msg.role === 'user'" class="flex justify-end">
            <div class="chat-bubble-user max-w-[80%] px-4 py-3 text-sm leading-relaxed text-gray-800">
              {{ msg.content }}
            </div>
          </div>
          <!-- AI 消息 — 左对齐 -->
          <div v-else class="flex justify-start">
            <div class="chat-bubble-ai max-w-[85%] px-4 py-3 text-sm leading-relaxed text-gray-700">
              <div v-html="renderMarkdown(msg.content)"></div>
            </div>
          </div>
        </template>

        <!-- 流式输出中 — 打字机效果 -->
        <div v-if="streaming" class="flex justify-start">
          <div class="chat-bubble-ai max-w-[85%] px-4 py-3 text-sm leading-relaxed text-gray-700">
            <div v-html="renderMarkdown(streamBuffer)"></div>
            <span class="typing-cursor"></span>
          </div>
        </div>
      </div>

      <!-- 底部输入区 -->
      <div class="px-5 py-4 border-t border-gray-100 flex-shrink-0 space-y-2">
        <!-- 划选文本引用提示 -->
        <div
          v-if="selectedText"
          class="px-3 py-2 bg-amber-50 rounded-xl text-xs text-amber-700 flex items-center gap-2"
        >
          <span class="truncate flex-1">引用: 「{{ selectedText }}」</span>
          <button @click="selectedText = ''" class="text-amber-500 hover:text-amber-700 text-base leading-none">&times;</button>
        </div>
        <!-- 输入框 + 发送按钮 -->
        <div class="flex gap-2.5">
          <input
            v-model="inputText"
            @keydown.enter="sendMessage(inputText)"
            placeholder="输入问题，按 Enter 发送..."
            :disabled="streaming"
            class="flex-1 px-4 py-3 text-sm bg-[#F9F9F9] border-none rounded-xl focus:outline-none focus:ring-2 focus:ring-amber-500/20 placeholder-gray-400 disabled:opacity-50"
          />
          <button
            @click="sendMessage(inputText)"
            :disabled="!inputText.trim() || streaming"
            class="px-4 py-3 bg-[#1A1A1A] text-white rounded-xl hover:bg-[#333] disabled:opacity-30 transition-colors flex-shrink-0"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
/**
 * AnalyzeMode.vue — 分析模式
 *
 * SSE 流式对话核心流程：
 * 1. 用户输入问题 → 调用 analyzeChat()
 * 2. fetch POST /api/chat → 返回 text/event-stream
 * 3. ReadableStream 逐块读取：
 *    - data: {"text": "..."} → 追加到 streamBuffer（实时渲染打字机效果）
 *    - data: {"done": true}  → 完成，将 streamBuffer 移入 messages
 *    - data: {"error": "..."} → 显示错误
 * 4. Markdown 渲染：使用 marked 库解析 AI 回答中的格式化内容
 */
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { analyzeChat } from '../api'
import { marked } from 'marked'
import * as pdfjsLib from 'pdfjs-dist'

pdfjsLib.GlobalWorkerOptions.workerSrc =
  `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

const props = defineProps({
  filename: String,
  paperData: Object,
})

// DOM refs
const pdfCanvas = ref(null)
const chatContainer = ref(null)

// PDF 状态
const currentPage = ref(1)
const totalPages = ref(0)
let pdfDoc = null

// 聊天状态
const inputText = ref('')
const selectedText = ref('')       // 用户在 PDF 上划选的文本
const messages = ref([])           // 历史消息 [{ role: 'user'|'assistant', content: '' }]
const streaming = ref(false)       // SSE 流式输出中
const streamBuffer = ref('')       // 当前正在流式接收的文本缓冲

// 推荐问题
const suggestedQuestions = [
  '请总结这篇论文的核心贡献和创新点',
  '论文使用了什么研究方法和技术路线？',
  '实验结果与 baseline 对比如何？',
  '这篇论文存在哪些局限性和未来改进方向？',
]

/** 渲染 Markdown（AI 回答中可能包含标题、列表、代码块等） */
function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true })
}

// ---- PDF.js 渲染（与 ReadMode 相同逻辑） ----

async function loadPdf() {
  if (!props.filename) return
  try {
    pdfDoc = await pdfjsLib.getDocument(`/uploads/${props.filename}`).promise
    totalPages.value = pdfDoc.numPages
    await renderPage(currentPage.value)
  } catch (err) {
    console.error('PDF 加载失败:', err)
  }
}

async function renderPage(pageNum) {
  if (!pdfDoc || !pdfCanvas.value) return
  const page = await pdfDoc.getPage(pageNum)
  const viewport = page.getViewport({ scale: 1.5 })
  const canvas = pdfCanvas.value
  canvas.width = viewport.width
  canvas.height = viewport.height
  await page.render({
    canvasContext: canvas.getContext('2d'),
    viewport,
  }).promise
}

async function goToPage(pageNum) {
  if (pageNum < 1 || pageNum > totalPages.value) return
  currentPage.value = pageNum
  await renderPage(pageNum)
}

// ---- SSE 流式对话 ----

/**
 * 发送消息并流式接收 AI 回答
 *
 * 调用 analyzeChat() 时传入 4 个回调：
 * - onChunk: 每收到一个文本片段 → 追加到 streamBuffer → 打字机效果
 * - onDone:  AI 回答完毕 → streamBuffer 内容移入 messages 数组
 * - onError: 出错 → 显示错误信息
 */
async function sendMessage(text) {
  if (!text?.trim() || streaming.value) return

  const question = text.trim()
  inputText.value = ''

  // 添加用户消息到列表
  messages.value.push({ role: 'user', content: question })

  // 构建 Gemini 多轮对话历史
  const chatHistory = messages.value.slice(0, -1).map(m => ({
    role: m.role === 'user' ? 'user' : 'model',
    parts: [m.content],
  }))

  streaming.value = true
  streamBuffer.value = ''
  await nextTick()
  scrollToBottom()

  // 调用 SSE 流式接口
  analyzeChat(
    {
      filename: props.filename,
      question,
      selected_text: selectedText.value || null,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
    },
    // onChunk — 实时追加文本（打字机效果）
    (chunk) => {
      streamBuffer.value += chunk
      nextTick(scrollToBottom)
    },
    // onDone — 流式完成，移入消息列表
    () => {
      messages.value.push({ role: 'assistant', content: streamBuffer.value })
      streamBuffer.value = ''
      streaming.value = false
      selectedText.value = ''
    },
    // onError
    (error) => {
      messages.value.push({ role: 'assistant', content: `[错误] ${error}` })
      streaming.value = false
    },
  )
}

/** 自动滚动到聊天区域底部 */
function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

/** 监听鼠标松开事件 — 捕获用户在 PDF 上的文本划选 */
function handleMouseUp() {
  const sel = window.getSelection()
  const text = sel?.toString().trim()
  if (text && text.length > 2) {
    selectedText.value = text
  }
}

// ---- 生命周期 ----

watch(() => props.filename, () => {
  messages.value = []
  currentPage.value = 1
  loadPdf()
})

onMounted(() => {
  if (props.filename) loadPdf()
  document.addEventListener('mouseup', handleMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mouseup', handleMouseUp)
})
</script>
