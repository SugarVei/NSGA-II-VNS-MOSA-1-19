<template>
  <div class="h-full flex">
    <!-- 左侧：原生 PDF 渲染 -->
    <div
      ref="pdfContainer"
      class="flex-1 overflow-y-auto bg-gray-100 p-4"
      @scroll="syncScroll"
    >
      <div class="max-w-3xl mx-auto space-y-2">
        <!-- 分页导航 -->
        <div class="flex items-center justify-center gap-4 py-2 sticky top-0 z-10 bg-gray-100/80 backdrop-blur-sm rounded-xl">
          <button
            @click="goToPage(currentPage - 1)"
            :disabled="currentPage <= 1"
            class="px-3 py-1.5 text-sm rounded-lg bg-claude-surface border border-claude-border hover:bg-claude-hover disabled:opacity-40 transition-colors"
          >
            上一页
          </button>
          <span class="text-sm text-claude-text-secondary">
            {{ currentPage }} / {{ totalPages }}
          </span>
          <button
            @click="goToPage(currentPage + 1)"
            :disabled="currentPage >= totalPages"
            class="px-3 py-1.5 text-sm rounded-lg bg-claude-surface border border-claude-border hover:bg-claude-hover disabled:opacity-40 transition-colors"
          >
            下一页
          </button>
          <button
            v-if="!translations[currentPage]"
            @click="translateCurrentPage"
            :disabled="translating"
            class="px-4 py-1.5 text-sm rounded-lg bg-claude-text text-white hover:bg-gray-800 disabled:opacity-50 transition-colors"
          >
            {{ translating ? '翻译中...' : '翻译本页' }}
          </button>
        </div>

        <!-- PDF 页面渲染 canvas -->
        <div class="bg-white rounded-xl shadow-sm overflow-hidden">
          <canvas ref="pdfCanvas" class="pdf-page-canvas w-full"></canvas>
        </div>
      </div>
    </div>

    <!-- 右侧：中文译文层（基于坐标绝对定位） -->
    <div
      ref="translationContainer"
      class="flex-1 overflow-y-auto bg-claude-bg p-4"
    >
      <div class="max-w-3xl mx-auto">
        <!-- 未翻译提示 -->
        <div v-if="!translations[currentPage]" class="h-full flex items-center justify-center">
          <div class="text-center text-claude-text-secondary space-y-2">
            <p class="text-sm">点击「翻译本页」查看中文译文</p>
            <p class="text-xs">译文将按段落坐标精准对齐</p>
          </div>
        </div>

        <!-- 译文渲染（坐标定位） -->
        <div
          v-else
          class="translation-overlay"
          :style="{ height: pageHeight + 'px', width: pageWidth + 'px' }"
        >
          <div
            v-for="block in translations[currentPage]"
            :key="block.block_id"
            class="translation-block"
            :style="{
              top: scaleY(block.bbox.y0) + 'px',
              left: scaleX(block.bbox.x0) + 'px',
              width: scaleX(block.bbox.x1 - block.bbox.x0) + 'px',
            }"
          >
            {{ block.translated }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import { translatePage } from '../api'
import * as pdfjsLib from 'pdfjs-dist'

// 设置 PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

const props = defineProps({
  filename: String,
  paperData: Object,
})

const pdfContainer = ref(null)
const translationContainer = ref(null)
const pdfCanvas = ref(null)

const currentPage = ref(1)
const totalPages = ref(0)
const translating = ref(false)
const translations = ref({})  // { pageNumber: [translatedBlocks] }
const pageWidth = ref(0)
const pageHeight = ref(0)

// PDF.js 文档实例
let pdfDoc = null
// 当前页面的原始 PDF 尺寸（用于坐标缩放）
let originalWidth = 0
let originalHeight = 0
// 实际渲染尺寸
let renderedWidth = 0
let renderedHeight = 0

/** 坐标缩放：将 PDF 原始坐标映射到实际渲染尺寸 */
function scaleX(x) {
  return originalWidth > 0 ? (x / originalWidth) * renderedWidth : x
}
function scaleY(y) {
  return originalHeight > 0 ? (y / originalHeight) * renderedHeight : y
}

/** 加载 PDF 文档 */
async function loadPdf() {
  if (!props.filename) return

  const url = `/api/papers/pdf/${props.filename}`
  // 尝试从后端获取 PDF；如果没有专用路由，用 uploads 路径
  try {
    pdfDoc = await pdfjsLib.getDocument(`/uploads/${props.filename}`).promise
  } catch {
    // 备选：通过后端代理
    pdfDoc = await pdfjsLib.getDocument(url).promise
  }
  totalPages.value = pdfDoc.numPages
  await renderPage(currentPage.value)
}

/** 渲染指定页面到 canvas */
async function renderPage(pageNum) {
  if (!pdfDoc) return

  const page = await pdfDoc.getPage(pageNum)
  const viewport = page.getViewport({ scale: 1.5 })

  originalWidth = page.getViewport({ scale: 1 }).width
  originalHeight = page.getViewport({ scale: 1 }).height
  renderedWidth = viewport.width
  renderedHeight = viewport.height

  pageWidth.value = viewport.width
  pageHeight.value = viewport.height

  const canvas = pdfCanvas.value
  canvas.width = viewport.width
  canvas.height = viewport.height
  const ctx = canvas.getContext('2d')

  await page.render({ canvasContext: ctx, viewport }).promise
}

/** 翻页 */
async function goToPage(page) {
  if (page < 1 || page > totalPages.value) return
  currentPage.value = page
  await renderPage(page)
}

/** 翻译当前页面 */
async function translateCurrentPage() {
  translating.value = true
  try {
    const result = await translatePage(props.filename, currentPage.value)
    translations.value[currentPage.value] = result.translations
  } catch (err) {
    console.error('翻译失败:', err)
  } finally {
    translating.value = false
  }
}

/** 滚动同步：左右面板联动 */
function syncScroll() {
  if (!pdfContainer.value || !translationContainer.value) return
  const ratio = pdfContainer.value.scrollTop / (pdfContainer.value.scrollHeight - pdfContainer.value.clientHeight || 1)
  translationContainer.value.scrollTop = ratio * (translationContainer.value.scrollHeight - translationContainer.value.clientHeight)
}

watch(() => props.filename, () => {
  translations.value = {}
  currentPage.value = 1
  loadPdf()
})

onMounted(() => {
  if (props.filename) loadPdf()
})
</script>
