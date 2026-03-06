<template>
  <div class="h-screen flex overflow-hidden">
    <!-- 可折叠侧边栏 -->
    <Sidebar
      v-show="sidebarOpen"
      :papers="papers"
      :current-paper="currentPaper"
      @select-paper="selectPaper"
      @upload="handleUpload"
      @delete="handleDelete"
      @config-saved="loadConfig"
      class="w-72 flex-shrink-0"
    />

    <!-- 主工作区 -->
    <div class="flex-1 flex flex-col min-w-0">
      <!-- 顶部导航栏 -->
      <header class="h-14 border-b border-claude-border bg-claude-surface flex items-center px-4 gap-4">
        <!-- 侧边栏切换按钮 -->
        <button
          @click="sidebarOpen = !sidebarOpen"
          class="p-2 rounded-lg hover:bg-claude-hover transition-colors"
          :title="sidebarOpen ? '收起侧边栏' : '展开侧边栏'"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"/>
          </svg>
        </button>

        <!-- 当前文件名 -->
        <span class="text-sm text-claude-text-secondary truncate">
          {{ currentPaper || '请从侧边栏选择或上传文献' }}
        </span>

        <!-- 模式切换 Tab -->
        <div v-if="currentPaper" class="ml-auto flex bg-claude-bg rounded-xl p-1">
          <button
            @click="mode = 'read'"
            :class="[
              'px-4 py-1.5 text-sm rounded-lg transition-all',
              mode === 'read'
                ? 'bg-claude-surface shadow-sm text-claude-text font-medium'
                : 'text-claude-text-secondary hover:text-claude-text'
            ]"
          >
            阅读模式
          </button>
          <button
            @click="mode = 'analyze'"
            :class="[
              'px-4 py-1.5 text-sm rounded-lg transition-all',
              mode === 'analyze'
                ? 'bg-claude-surface shadow-sm text-claude-text font-medium'
                : 'text-claude-text-secondary hover:text-claude-text'
            ]"
          >
            分析模式
          </button>
        </div>
      </header>

      <!-- 内容区 -->
      <main class="flex-1 overflow-hidden">
        <!-- 未选择文件时的欢迎页 -->
        <div v-if="!currentPaper" class="h-full flex items-center justify-center">
          <div class="text-center space-y-4">
            <div class="w-16 h-16 mx-auto rounded-2xl bg-claude-hover flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-8 h-8 text-claude-text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
              </svg>
            </div>
            <h2 class="text-xl font-medium text-claude-text">Paper-Reader-Pro</h2>
            <p class="text-claude-text-secondary text-sm">上传 PDF 文献，开始 AI 辅助阅读</p>
          </div>
        </div>

        <!-- 阅读模式：双语分屏 -->
        <ReadMode
          v-else-if="mode === 'read'"
          :filename="currentPaper"
          :paper-data="paperData"
        />

        <!-- 分析模式：左文右聊 -->
        <AnalyzeMode
          v-else
          :filename="currentPaper"
          :paper-data="paperData"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { listPapers, uploadAndParse, deletePaper, getConfig } from './api'
import Sidebar from './components/Sidebar.vue'
import ReadMode from './views/ReadMode.vue'
import AnalyzeMode from './views/AnalyzeMode.vue'

const sidebarOpen = ref(true)
const mode = ref('read')           // 'read' | 'analyze'
const papers = ref([])             // 文献列表
const currentPaper = ref(null)     // 当前选中的文献文件名
const paperData = ref(null)        // 当前文献的解析数据（含坐标+译文）
const uploading = ref(false)       // 上传+解析+翻译中

/** 加载文献列表 */
async function loadPapers() {
  try {
    papers.value = await listPapers()
  } catch { /* 后端未启动时忽略 */ }
}

/** 加载配置状态 */
async function loadConfig() {
  try { await getConfig() } catch {}
}

/** 选择已有文献（从列表中点击） */
async function selectPaper(filename) {
  currentPaper.value = filename
  mode.value = 'read'
  // 已上传的文件需要重新调用 upload_and_parse 获取数据
  // 或者如果之前已有缓存数据则直接使用
}

/**
 * 上传文件 → 调用 /api/upload_and_parse 一站式处理
 * 返回值直接包含坐标 + 译文，无需再单独调翻译接口
 */
async function handleUpload(file) {
  uploading.value = true
  try {
    const result = await uploadAndParse(file)
    paperData.value = result
    currentPaper.value = result.filename
    mode.value = 'read'
    await loadPapers()
  } catch (err) {
    console.error('上传解析失败:', err)
  } finally {
    uploading.value = false
  }
}

/** 删除文献 */
async function handleDelete(filename) {
  await deletePaper(filename)
  if (currentPaper.value === filename) {
    currentPaper.value = null
    paperData.value = null
  }
  await loadPapers()
}

onMounted(() => {
  loadPapers()
  loadConfig()
})
</script>
