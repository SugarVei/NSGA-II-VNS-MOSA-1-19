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
        <!-- 没有数据时的提示 -->
        <div v-if="!currentPageData" class="h-full flex items-center justify-center">
          <div class="text-center text-claude-text-secondary space-y-2">
            <p class="text-sm">上传 PDF 后自动显示中文译文</p>
            <p class="text-xs">译文按段落坐标精准对齐</p>
          </div>
        </div>

        <!-- 译文渲染（基于 BBox 坐标绝对定位） -->
        <div
          v-else
          class="translation-overlay"
          :style="{ height: pageHeight + 'px', width: pageWidth + 'px' }"
        >
          <div
            v-for="block in translatableBlocks"
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
import { ref, computed, watch, onMounted } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'

// 设置 PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

const props = defineProps({
  filename: String,
  // paperData 来自 /api/upload_and_parse 的返回值，已包含坐标 + 译文
  paperData: Object,
})

const pdfContainer = ref(null)
const translationContainer = ref(null)
const pdfCanvas = ref(null)

const currentPage = ref(1)
const totalPages = ref(0)
const pageWidth = ref(0)
const pageHeight = ref(0)

let pdfDoc = null
let originalWidth = 0
let originalHeight = 0
let renderedWidth = 0
let renderedHeight = 0

/** 当前页的数据（从 paperData 中提取） */
const currentPageData = computed(() => {
  if (!props.paperData?.pages) return null
  return props.paperData.pages.find(p => p.page_number === currentPage.value)
})

/** 当前页中有译文的文本块 */
const translatableBlocks = computed(() => {
  if (!currentPageData.value) return []
  return currentPageData.value.blocks.filter(b => b.is_translatable && b.translated)
})

/** 坐标缩放：PDF 原始坐标 → 实际渲染像素 */
function scaleX(x) {
  return originalWidth > 0 ? (x / originalWidth) * renderedWidth : x
}
function scaleY(y) {
  return originalHeight > 0 ? (y / originalHeight) * renderedHeight : y
}

/** 加载 PDF 文档 */
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

  await page.render({
    canvasContext: canvas.getContext('2d'),
    viewport,
  }).promise
}

/** 翻页 */
async function goToPage(page) {
  if (page < 1 || page > totalPages.value) return
  currentPage.value = page
  await renderPage(page)
}

/** 滚动同步：左右面板联动 */
function syncScroll() {
  if (!pdfContainer.value || !translationContainer.value) return
  const scrollable = pdfContainer.value.scrollHeight - pdfContainer.value.clientHeight
  const ratio = scrollable > 0 ? pdfContainer.value.scrollTop / scrollable : 0
  const targetScrollable = translationContainer.value.scrollHeight - translationContainer.value.clientHeight
  translationContainer.value.scrollTop = ratio * targetScrollable
}

watch(() => props.filename, () => {
  currentPage.value = 1
  loadPdf()
})

onMounted(() => {
  if (props.filename) loadPdf()
})
</script>
