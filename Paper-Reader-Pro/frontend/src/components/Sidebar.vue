<template>
  <aside class="h-full bg-claude-surface border-r border-claude-border flex flex-col">
    <!-- 标题 -->
    <div class="p-4 border-b border-claude-border">
      <h1 class="text-base font-semibold tracking-tight">Paper-Reader-Pro</h1>
    </div>

    <!-- 文献书架 -->
    <div class="flex-1 overflow-y-auto p-3 space-y-1">
      <div class="text-xs font-medium text-claude-text-secondary uppercase tracking-wider px-2 py-2">
        文献书架
      </div>

      <!-- 拖拽上传区域 -->
      <div
        @drop.prevent="onDrop"
        @dragover.prevent="dragOver = true"
        @dragleave="dragOver = false"
        @click="triggerFileInput"
        :class="[
          'border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors text-sm',
          dragOver
            ? 'border-claude-accent bg-amber-50'
            : 'border-claude-border hover:border-claude-text-secondary'
        ]"
      >
        <p class="text-claude-text-secondary">
          拖拽 PDF 到此处<br/>或点击上传
        </p>
        <input
          ref="fileInput"
          type="file"
          accept=".pdf"
          class="hidden"
          @change="onFileSelect"
        />
      </div>

      <!-- 文献列表 -->
      <div
        v-for="paper in papers"
        :key="paper"
        @click="$emit('selectPaper', paper)"
        :class="[
          'flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer transition-colors group',
          currentPaper === paper ? 'bg-claude-hover' : 'hover:bg-claude-hover'
        ]"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 text-claude-text-secondary flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
        </svg>
        <span class="text-sm truncate flex-1">{{ paper }}</span>
        <button
          @click.stop="$emit('delete', paper)"
          class="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 text-claude-text-secondary hover:text-red-500 transition-all"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- API 配置中心 -->
    <div class="border-t border-claude-border p-3 space-y-3">
      <div class="text-xs font-medium text-claude-text-secondary uppercase tracking-wider px-2">
        API 配置
      </div>
      <div class="space-y-2">
        <input
          v-model="openaiKey"
          type="password"
          placeholder="OpenAI API Key"
          class="w-full px-3 py-2 text-sm bg-claude-bg border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent/30 focus:border-claude-accent"
        />
        <input
          v-model="geminiKey"
          type="password"
          placeholder="Gemini API Key"
          class="w-full px-3 py-2 text-sm bg-claude-bg border border-claude-border rounded-lg focus:outline-none focus:ring-2 focus:ring-claude-accent/30 focus:border-claude-accent"
        />
        <button
          @click="saveKeys"
          class="w-full py-2 text-sm bg-claude-text text-white rounded-lg hover:bg-gray-800 transition-colors"
        >
          保存配置
        </button>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { saveConfig } from '../api'

const props = defineProps({
  papers: Array,
  currentPaper: String,
})

const emit = defineEmits(['selectPaper', 'upload', 'delete', 'configSaved'])

const fileInput = ref(null)
const dragOver = ref(false)

// API Key — 从 localStorage 恢复
const openaiKey = ref(localStorage.getItem('openai_key') || '')
const geminiKey = ref(localStorage.getItem('gemini_key') || '')

function triggerFileInput() {
  fileInput.value?.click()
}

function onFileSelect(e) {
  const file = e.target.files[0]
  if (file) emit('upload', file)
  e.target.value = '' // 重置，允许重复上传同名文件
}

function onDrop(e) {
  dragOver.value = false
  const file = e.dataTransfer.files[0]
  if (file && file.name.endsWith('.pdf')) {
    emit('upload', file)
  }
}

async function saveKeys() {
  // 持久化到 localStorage
  localStorage.setItem('openai_key', openaiKey.value)
  localStorage.setItem('gemini_key', geminiKey.value)
  // 同步到后端运行时
  await saveConfig({
    openai_key: openaiKey.value || null,
    gemini_key: geminiKey.value || null,
  })
  emit('configSaved')
}

// 页面加载时自动同步已保存的 Key 到后端
onMounted(async () => {
  if (openaiKey.value || geminiKey.value) {
    try {
      await saveConfig({
        openai_key: openaiKey.value || null,
        gemini_key: geminiKey.value || null,
      })
    } catch { /* 后端未启动时忽略 */ }
  }
})
</script>
