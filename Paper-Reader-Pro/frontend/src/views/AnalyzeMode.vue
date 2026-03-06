<template>
  <div class="h-full flex">
    <!-- 左侧：PDF 阅读视图 -->
    <div class="flex-1 overflow-y-auto bg-gray-100 p-4">
      <div class="max-w-3xl mx-auto">
        <div class="bg-white rounded-xl shadow-sm overflow-hidden">
          <canvas ref="pdfCanvas" class="pdf-page-canvas w-full"></canvas>
        </div>
        <!-- 分页控制 -->
        <div class="flex items-center justify-center gap-4 py-3">
          <button
            @click="goToPage(currentPage - 1)"
            :disabled="currentPage <= 1"
            class="px-3 py-1.5 text-sm rounded-lg bg-claude-surface border border-claude-border hover:bg-claude-hover disabled:opacity-40"
          >
            上一页
          </button>
          <span class="text-sm text-claude-text-secondary">{{ currentPage }} / {{ totalPages }}</span>
          <button
            @click="goToPage(currentPage + 1)"
            :disabled="currentPage >= totalPages"
            class="px-3 py-1.5 text-sm rounded-lg bg-claude-surface border border-claude-border hover:bg-claude-hover disabled:opacity-40"
          >
            下一页
          </button>
        </div>
      </div>
    </div>

    <!-- 右侧：Claude 风格对话窗口 -->
    <div class="w-[420px] flex-shrink-0 border-l border-claude-border bg-claude-surface flex flex-col">
      <!-- 聊天头部 -->
      <div class="px-4 py-3 border-b border-claude-border">
        <h3 class="text-sm font-medium">AI 论文分析</h3>
        <p class="text-xs text-claude-text-secondary mt-0.5">基于论文全文的智能问答</p>
      </div>

      <!-- 消息列表 -->
      <div ref="chatContainer" class="flex-1 overflow-y-auto p-4 space-y-4">
        <!-- 欢迎消息 -->
        <div v-if="messages.length === 0" class="text-center py-8">
          <p class="text-sm text-claude-text-secondary">向 AI 提问关于这篇论文的任何问题</p>
          <div class="mt-4 space-y-2">
            <button
              v-for="q in suggestedQuestions"
              :key="q"
              @click="sendMessage(q)"
              class="block w-full text-left px-3 py-2 text-sm rounded-xl border border-claude-border hover:bg-claude-hover transition-colors"
            >
              {{ q }}
            </button>
          </div>
        </div>

        <!-- 消息气泡 -->
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'"
        >
          <div
            :class="[
              'max-w-[85%] px-4 py-3 text-sm leading-relaxed',
              msg.role === 'user' ? 'chat-message-user' : 'chat-message-ai'
            ]"
          >
            <div v-html="renderMarkdown(msg.content)"></div>
          </div>
        </div>

        <!-- 流式输出中的加载指示器 -->
        <div v-if="streaming" class="flex justify-start">
          <div class="chat-message-ai max-w-[85%] px-4 py-3 text-sm">
            <div v-html="renderMarkdown(streamBuffer)"></div>
            <span class="inline-block w-1.5 h-4 bg-claude-accent animate-pulse ml-0.5"></span>
          </div>
        </div>
      </div>

      <!-- 输入框 -->
      <div class="p-4 border-t border-claude-border">
        <!-- 划选文本提示 -->
        <div
          v-if="selectedText"
          class="mb-2 px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-xs flex items-center gap-2"
        >
          <span class="truncate flex-1">引用: 「{{ selectedText }}」</span>
          <button @click="selectedText = ''" class="text-amber-600 hover:text-amber-800">&times;</button>
        </div>
        <div class="flex gap-2">
          <input
            v-model="inputText"
            @keydown.enter="sendMessage(inputText)"
            placeholder="输入问题..."
            :disabled="streaming"
            class="flex-1 px-4 py-2.5 text-sm bg-claude-bg border border-claude-border rounded-xl focus:outline-none focus:ring-2 focus:ring-claude-accent/30 focus:border-claude-accent"
          />
          <button
            @click="sendMessage(inputText)"
            :disabled="!inputText.trim() || streaming"
            class="px-4 py-2.5 bg-claude-text text-white rounded-xl hover:bg-gray-800 disabled:opacity-40 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import { analyzeChat } from '../api'
import { marked } from 'marked'
import * as pdfjsLib from 'pdfjs-dist'

pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

const props = defineProps({
  filename: String,
  paperData: Object,
})

const pdfCanvas = ref(null)
const chatContainer = ref(null)

const currentPage = ref(1)
const totalPages = ref(0)
const inputText = ref('')
const selectedText = ref('')
const messages = ref([])
const streaming = ref(false)
const streamBuffer = ref('')

let pdfDoc = null

const suggestedQuestions = [
  '请总结这篇论文的核心贡献',
  '这篇论文使用了什么方法论？',
  '论文的实验结果如何？',
  '这篇论文有哪些局限性？',
]

/** 渲染 Markdown */
function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text, { breaks: true })
}

/** 加载 PDF */
async function loadPdf() {
  if (!props.filename) return
  try {
    pdfDoc = await pdfjsLib.getDocument(`/uploads/${props.filename}`).promise
  } catch {
    pdfDoc = await pdfjsLib.getDocument(`/api/papers/pdf/${props.filename}`).promise
  }
  totalPages.value = pdfDoc.numPages
  await renderPage(currentPage.value)
}

async function renderPage(pageNum) {
  if (!pdfDoc) return
  const page = await pdfDoc.getPage(pageNum)
  const viewport = page.getViewport({ scale: 1.5 })
  const canvas = pdfCanvas.value
  canvas.width = viewport.width
  canvas.height = viewport.height
  await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise
}

async function goToPage(page) {
  if (page < 1 || page > totalPages.value) return
  currentPage.value = page
  await renderPage(page)
}

/** 发送消息（SSE 流式接收） */
async function sendMessage(text) {
  if (!text?.trim() || streaming.value) return

  const question = text.trim()
  inputText.value = ''

  // 添加用户消息
  messages.value.push({ role: 'user', content: question })

  // 构建历史对话（供 Gemini 多轮对话）
  const chatHistory = messages.value.slice(0, -1).map(m => ({
    role: m.role === 'user' ? 'user' : 'model',
    parts: [m.content],
  }))

  streaming.value = true
  streamBuffer.value = ''

  await nextTick()
  scrollToBottom()

  analyzeChat(
    {
      filename: props.filename,
      question,
      selected_text: selectedText.value || null,
      chat_history: chatHistory.length > 0 ? chatHistory : null,
    },
    // onChunk: 逐块追加到缓冲区
    (chunk) => {
      streamBuffer.value += chunk
      nextTick(scrollToBottom)
    },
    // onDone: 将完整回答添加到消息列表
    () => {
      messages.value.push({ role: 'assistant', content: streamBuffer.value })
      streamBuffer.value = ''
      streaming.value = false
      selectedText.value = ''
    },
    // onError
    (error) => {
      messages.value.push({ role: 'assistant', content: `错误: ${error}` })
      streaming.value = false
    },
  )
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  }
}

// 监听 PDF 文本选择（用于划选发送到聊天）
function handleTextSelection() {
  const selection = window.getSelection()
  if (selection && selection.toString().trim()) {
    selectedText.value = selection.toString().trim()
  }
}

watch(() => props.filename, () => {
  messages.value = []
  currentPage.value = 1
  loadPdf()
})

onMounted(() => {
  if (props.filename) loadPdf()
  document.addEventListener('mouseup', handleTextSelection)
})
</script>
