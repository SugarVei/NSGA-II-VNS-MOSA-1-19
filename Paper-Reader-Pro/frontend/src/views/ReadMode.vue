<template>
  <!--
    阅读模式 — 双语精准对齐视图
    =============================
    左右各占 50%（w-1/2），中间 gap-4，白色圆角卡片。

    核心技术：
    1. 左侧 PDF.js 渲染到 Canvas，获取原始/渲染尺寸
    2. scale = renderedCanvasWidth / pdfOriginalPageWidth
    3. 右侧容器等高等宽，遍历翻译块用 absolute 定位：
       top: y0*scale, left: x0*scale, width: (x1-x0)*scale
    4. 左侧 onScroll → 等比例同步右侧 scrollTop
  -->
  <div class="h-full flex gap-4">

    <!-- ========== 左半：英文原始 PDF ========== -->
    <div
      ref="leftPanel"
      class="w-1/2 overflow-y-auto rounded-2xl bg-white shadow-sm"
      @scroll="onLeftScroll"
    >
      <!-- 分页导航 — 吸顶 -->
      <div class="sticky top-0 z-10 flex items-center justify-center gap-4 py-3 bg-white/90 backdrop-blur-sm border-b border-gray-100">
        <button
          @click="goToPage(currentPage - 1)"
          :disabled="currentPage <= 1"
          class="px-3 py-1 text-sm rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
        >&larr; 上一页</button>
        <span class="text-sm text-gray-400 tabular-nums">{{ currentPage }} / {{ totalPages }}</span>
        <button
          @click="goToPage(currentPage + 1)"
          :disabled="currentPage >= totalPages"
          class="px-3 py-1 text-sm rounded-lg hover:bg-gray-100 disabled:opacity-30 transition-colors"
        >下一页 &rarr;</button>
      </div>
      <!-- PDF Canvas -->
      <div class="p-4">
        <canvas ref="pdfCanvas" class="pdf-page-canvas w-full"></canvas>
      </div>
    </div>

    <!-- ========== 右半：中文译文（BBox 坐标绝对定位） ========== -->
    <div
      ref="rightPanel"
      class="w-1/2 overflow-y-auto rounded-2xl bg-white shadow-sm"
    >
      <div class="sticky top-0 z-10 flex items-center justify-center py-3 bg-white/90 backdrop-blur-sm border-b border-gray-100">
        <span class="text-sm text-gray-400">中文译文</span>
      </div>

      <!-- 无数据提示 -->
      <div v-if="!hasTranslation" class="flex items-center justify-center h-[calc(100%-48px)]">
        <div class="text-center text-gray-300 space-y-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-12 h-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="0.8">
            <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 21l5.25-11.25L21 21m-9-3h7.5M3 5.621a48.474 48.474 0 016-.371m0 0c1.12 0 2.233.038 3.334.114M9 5.25V3m3.334 2.364C11.176 10.658 7.69 15.08 3 17.502m9.334-12.138c.896.061 1.785.147 2.666.257m-4.589 8.495a18.023 18.023 0 01-3.827-5.802"/>
          </svg>
          <p class="text-sm">上传 PDF 后自动显示翻译</p>
          <p class="text-xs">每段译文精准对齐至原文位置</p>
        </div>
      </div>

      <!--
        译文渲染层 — 核心实现
        容器宽高 = Canvas 像素尺寸（与左侧 PDF 完全等高等宽）
        每个 .translation-block 通过 absolute 定位到原文对应坐标
      -->
      <div
        v-else
        class="translation-overlay p-4"
        :style="{
          width: renderedWidth + 'px',
          height: renderedHeight + 'px',
        }"
      >
        <div
          v-for="block in currentTranslatedBlocks"
          :key="block.block_id"
          class="translation-block"
          :style="{
            top:   (block.bbox.y0 * scale) + 'px',
            left:  (block.bbox.x0 * scale) + 'px',
            width: ((block.bbox.x1 - block.bbox.x0) * scale) + 'px',
          }"
        >
          {{ block.translated }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
/**
 * ReadMode.vue — 阅读模式核心组件
 *
 * 坐标转换公式：
 *   scale = renderedCanvasWidth / pdfOriginalPageWidth
 *   screenY = bbox.y0 * scale
 *   screenX = bbox.x0 * scale
 *   screenW = (bbox.x1 - bbox.x0) * scale
 *
 * 后端 bbox 单位 = PDF point (1pt = 1/72 inch)
 * PDF.js viewport({ scale:1 }).width = 原始 page point 宽度
 * 所以两者单位一致，乘以同一个 scale 即可映射到屏幕像素
 */
import { ref, computed, watch, onMounted } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'

pdfjsLib.GlobalWorkerOptions.workerSrc =
  `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

const props = defineProps({
  filename: String,
  paperData: Object,  // /api/upload_and_parse 返回的完整数据
})

// DOM refs
const leftPanel = ref(null)
const rightPanel = ref(null)
const pdfCanvas = ref(null)

// 页面状态
const currentPage = ref(1)
const totalPages = ref(0)

// 缩放与尺寸
const scale = ref(1)           // 坐标缩放比
const renderedWidth = ref(0)   // Canvas 实际像素宽
const renderedHeight = ref(0)  // Canvas 实际像素高

let pdfDoc = null

/** 当前页有译文的文本块 */
const currentTranslatedBlocks = computed(() => {
  if (!props.paperData?.pages) return []
  const page = props.paperData.pages.find(p => p.page_number === currentPage.value)
  if (!page) return []
  return page.blocks.filter(b => b.is_translatable && b.translated)
})

const hasTranslation = computed(() => currentTranslatedBlocks.value.length > 0)

// ---- PDF.js 渲染 ----

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

/**
 * 渲染指定页到 Canvas，并计算坐标缩放比
 *
 * 1. viewport({ scale:1 }) → 原始 PDF 尺寸（与后端 bbox 单位一致）
 * 2. viewport({ scale:1.5 }) → 放大渲染（清晰度）
 * 3. scale = rendered / original → 供右侧译文坐标转换
 */
async function renderPage(pageNum) {
  if (!pdfDoc || !pdfCanvas.value) return

  const page = await pdfDoc.getPage(pageNum)
  const originalVP = page.getViewport({ scale: 1 })
  const renderVP = page.getViewport({ scale: 1.5 })

  const canvas = pdfCanvas.value
  canvas.width = renderVP.width
  canvas.height = renderVP.height

  // 核心：计算坐标缩放比
  scale.value = renderVP.width / originalVP.width
  renderedWidth.value = renderVP.width
  renderedHeight.value = renderVP.height

  await page.render({
    canvasContext: canvas.getContext('2d'),
    viewport: renderVP,
  }).promise
}

async function goToPage(pageNum) {
  if (pageNum < 1 || pageNum > totalPages.value) return
  currentPage.value = pageNum
  await renderPage(pageNum)
  if (leftPanel.value) leftPanel.value.scrollTop = 0
  if (rightPanel.value) rightPanel.value.scrollTop = 0
}

// ---- 滚动联动 ----

function onLeftScroll() {
  if (!leftPanel.value || !rightPanel.value) return
  const maxL = leftPanel.value.scrollHeight - leftPanel.value.clientHeight
  if (maxL <= 0) return
  const ratio = leftPanel.value.scrollTop / maxL
  const maxR = rightPanel.value.scrollHeight - rightPanel.value.clientHeight
  rightPanel.value.scrollTop = ratio * maxR
}

// ---- 生命周期 ----

watch(() => props.filename, () => {
  currentPage.value = 1
  loadPdf()
})

onMounted(() => {
  if (props.filename) loadPdf()
})
</script>
