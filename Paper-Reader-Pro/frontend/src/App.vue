<template>
  <!-- 全屏容器：柔和灰色背景 -->
  <div class="h-screen w-screen bg-[#F9F9F9] flex flex-col overflow-hidden">

    <!-- ============================================================
         顶部导航栏 — 极简设计，居中 Tab + 右侧齿轮
         ============================================================ -->
    <header class="h-14 flex items-center justify-between px-6 flex-shrink-0">
      <!-- 左侧：Logo + 文件名 -->
      <div class="flex items-center gap-3 min-w-0">
        <h1 class="text-base font-semibold tracking-tight whitespace-nowrap">Paper-Reader-Pro</h1>
        <span v-if="currentPaper" class="text-sm text-gray-400 truncate max-w-[200px]">
          / {{ currentPaper }}
        </span>
      </div>

      <!-- 中间：模式切换 Tab (Switcher) -->
      <div
        v-if="currentPaper"
        class="absolute left-1/2 -translate-x-1/2 flex bg-[#F0F0F0] rounded-xl p-1"
      >
        <button
          @click="mode = 'read'"
          :class="['px-5 py-1.5 text-sm rounded-lg transition-all duration-200',
                    mode === 'read' ? 'tab-active' : 'tab-inactive']"
        >
          阅读模式
        </button>
        <button
          @click="mode = 'analyze'"
          :class="['px-5 py-1.5 text-sm rounded-lg transition-all duration-200',
                    mode === 'analyze' ? 'tab-active' : 'tab-inactive']"
        >
          分析模式
        </button>
      </div>

      <!-- 右侧：齿轮配置按钮 -->
      <button
        @click="showConfig = true"
        class="p-2 rounded-xl hover:bg-gray-200/60 transition-colors"
        title="API 配置"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>
    </header>

    <!-- ============================================================
         主工作区
         ============================================================ -->
    <main class="flex-1 overflow-hidden px-5 pb-5">
      <!-- 未选择文件时：欢迎页 + 上传区 -->
      <div v-if="!currentPaper" class="h-full flex items-center justify-center">
        <div class="text-center space-y-6 max-w-md">
          <!-- 图标 -->
          <div class="w-20 h-20 mx-auto rounded-3xl bg-white shadow-sm flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
            </svg>
          </div>
          <div>
            <h2 class="text-2xl font-semibold text-gray-800">开始阅读文献</h2>
            <p class="text-gray-400 text-sm mt-2">上传一篇 PDF 论文，AI 将自动翻译并辅助你阅读</p>
          </div>

          <!-- 拖拽上传区 -->
          <div
            @drop.prevent="onDrop"
            @dragover.prevent="dragActive = true"
            @dragleave="dragActive = false"
            @click="$refs.fileInput.click()"
            :class="[
              'border-2 border-dashed rounded-2xl p-8 cursor-pointer transition-all duration-200',
              dragActive ? 'upload-zone-active' : 'border-gray-300 hover:border-gray-400'
            ]"
          >
            <p class="text-gray-400 text-sm">
              {{ uploading ? '正在上传并解析翻译...' : '拖拽 PDF 文件到此处，或点击选择' }}
            </p>
            <input ref="fileInput" type="file" accept=".pdf" class="hidden" @change="onFileSelect" />
          </div>

          <!-- 已有文献快速选择 -->
          <div v-if="papers.length > 0" class="space-y-2">
            <p class="text-xs text-gray-400 uppercase tracking-wider">已上传文献</p>
            <div class="flex flex-wrap justify-center gap-2">
              <button
                v-for="paper in papers"
                :key="paper"
                @click="selectPaper(paper)"
                class="px-3 py-1.5 text-sm bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow"
              >
                {{ paper }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- 已选文件：阅读模式 / 分析模式 -->
      <template v-else>
        <!-- 阅读模式 -->
        <ReadMode
          v-if="mode === 'read'"
          :filename="currentPaper"
          :paper-data="paperData"
        />
        <!-- 分析模式 -->
        <AnalyzeMode
          v-else
          :filename="currentPaper"
          :paper-data="paperData"
        />
      </template>
    </main>

    <!-- ============================================================
         API 配置弹窗 (Modal)
         ============================================================ -->
    <Teleport to="body">
      <div
        v-if="showConfig"
        class="fixed inset-0 z-50 flex items-center justify-center modal-backdrop"
        @click.self="showConfig = false"
      >
        <div class="bg-white rounded-2xl shadow-xl w-[420px] p-6 space-y-5">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold">API 配置</h3>
            <button @click="showConfig = false" class="p-1 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <!-- OpenAI Key -->
          <div class="space-y-1.5">
            <label class="text-sm font-medium text-gray-600">OpenAI API Key</label>
            <p class="text-xs text-gray-400">用于段落翻译 (gpt-4o-mini)</p>
            <input
              v-model="openaiKey"
              type="password"
              placeholder="sk-..."
              class="w-full px-4 py-2.5 text-sm bg-[#F9F9F9] border-none rounded-xl focus:outline-none focus:ring-2 focus:ring-amber-500/30"
            />
          </div>

          <!-- Gemini Key -->
          <div class="space-y-1.5">
            <label class="text-sm font-medium text-gray-600">Gemini API Key</label>
            <p class="text-xs text-gray-400">用于论文分析对话 (gemini-1.5-flash)</p>
            <input
              v-model="geminiKey"
              type="password"
              placeholder="AIza..."
              class="w-full px-4 py-2.5 text-sm bg-[#F9F9F9] border-none rounded-xl focus:outline-none focus:ring-2 focus:ring-amber-500/30"
            />
          </div>

          <button
            @click="saveApiKeys"
            class="w-full py-2.5 text-sm font-medium bg-[#1A1A1A] text-white rounded-xl hover:bg-[#333] transition-colors"
          >
            保存配置
          </button>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
/**
 * App.vue — 主入口组件
 *
 * 职责：
 * 1. 顶部导航：居中 Switcher（阅读/分析）+ 右侧齿轮（API配置弹窗）
 * 2. 欢迎页：拖拽上传 PDF + 已有文献列表
 * 3. 路由到 ReadMode / AnalyzeMode 子组件
 * 4. API Key 管理（localStorage 持久化 + 后端同步）
 */
import { ref, onMounted } from 'vue'
import { listPapers, uploadAndParse, saveConfig } from './api'
import ReadMode from './views/ReadMode.vue'
import AnalyzeMode from './views/AnalyzeMode.vue'

// ---- 状态 ----
const mode = ref('read')              // 'read' | 'analyze'
const papers = ref([])                // 已上传的文献文件名列表
const currentPaper = ref(null)        // 当前选中的文件名
const paperData = ref(null)           // /api/upload_and_parse 返回的完整数据
const uploading = ref(false)          // 上传+解析+翻译进行中
const showConfig = ref(false)         // 配置弹窗是否显示
const dragActive = ref(false)         // 拖拽区域是否激活

// API Key（从 localStorage 恢复）
const openaiKey = ref(localStorage.getItem('openai_key') || '')
const geminiKey = ref(localStorage.getItem('gemini_key') || '')

// ---- 文献列表 ----
async function loadPapers() {
  try { papers.value = await listPapers() } catch {}
}

// ---- 选择已有文献（暂无缓存的解析数据，只记录文件名） ----
function selectPaper(filename) {
  currentPaper.value = filename
  paperData.value = null  // 需要重新上传才能获得翻译数据
  mode.value = 'read'
}

// ---- 上传文件 → /api/upload_and_parse ----
async function handleUpload(file) {
  uploading.value = true
  try {
    const result = await uploadAndParse(file)
    paperData.value = result                // 包含坐标 + 译文的完整 JSON
    currentPaper.value = result.filename
    mode.value = 'read'
    await loadPapers()
  } catch (err) {
    console.error('上传解析失败:', err)
  } finally {
    uploading.value = false
  }
}

function onFileSelect(e) {
  const file = e.target.files?.[0]
  if (file) handleUpload(file)
  e.target.value = ''
}

function onDrop(e) {
  dragActive.value = false
  const file = e.dataTransfer.files?.[0]
  if (file?.name.toLowerCase().endsWith('.pdf')) handleUpload(file)
}

// ---- API Key 管理 ----
async function saveApiKeys() {
  localStorage.setItem('openai_key', openaiKey.value)
  localStorage.setItem('gemini_key', geminiKey.value)
  try {
    await saveConfig({
      openai_key: openaiKey.value || null,
      gemini_key: geminiKey.value || null,
    })
  } catch {}
  showConfig.value = false
}

// ---- 初始化 ----
onMounted(async () => {
  loadPapers()
  // 自动同步本地缓存的 Key 到后端
  if (openaiKey.value || geminiKey.value) {
    try {
      await saveConfig({
        openai_key: openaiKey.value || null,
        gemini_key: geminiKey.value || null,
      })
    } catch {}
  }
})
</script>
