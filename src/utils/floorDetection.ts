import * as ort from 'onnxruntime-web'

/** Model input size. Use 512 unless your ONNX model supports dynamic shape (e.g. 768/1024 for finer rug/floor edges). */
const IMAGE_SIZE = 512
const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]
const USE_BGR = false
const FLOOR_CLASS_INDEX = 3
/** ADE20K-style class IDs for furniture: exclude these from floor mask (chair, sofa, table, bed, desk, cabinet). */
const FURNITURE_CLASS_INDICES = new Set([8, 9, 11, 12, 13, 14])
/** Subtract furniture pixels from floor mask when model predicts them. */
const SUBTRACT_FURNITURE_FROM_FLOOR = true
const BINARY_THRESHOLD = 0.5
/** Remove small floor patches. Higher (0.006) = remove areas under legs, thin bridges to furniture. */
const MIN_COMPONENT_AREA_RATIO = 0.006
/** true = use softmax threshold; false = argmax only. */
const USE_SOFTMAX_THRESHOLD = true
/** Include pixel as floor if P(floor) >= this. Higher = exclude furniture legs, only high-confidence floor. */
const FLOOR_PROB_THRESHOLD = 0.44
/** Use AND: floor only when (argmax=floor) AND (P(floor)>=threshold). Excludes low-confidence areas under legs. */
const USE_HYBRID_AND = true
const CONTRAST_STRETCH = true
/** 3×3 close: kichik teshiklarni yopadi, chegarani haddan ortiq kengaytirmaydi. 2 = 5×5 (teparoqga chiqib ketadi). */
const MORPH_CLOSE_RADIUS = 1
/** Erosion iterations: shrink floor mask to remove thin bridges under furniture legs. */
const ERODE_ITERATIONS = 1
/** Opening: 1 dilation after erosion restores main floor while keeping thin bridges removed. */
const EDGE_DILATE_ITERATIONS = 1
/** Vertical dilation o‘chirildi: pol yuqoriga kengayib gilam tepasini kesmasin, devor/mebelga chiqmasin. */
const VERTICAL_DILATE_ITERATIONS = 0
/** Test-time augmentation: run on original + flipped, merge with AND for higher accuracy. */
const USE_TTA = false

let session: ort.InferenceSession | null = null

// CDN ishlatamiz
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

async function getSession() {
  if (session) return session
  console.log('[FloorDetection] Loading ONNX model...')
  session = await ort.InferenceSession.create('/models/model.onnx', {
    executionProviders: ['wasm'],
  })
  console.log('[FloorDetection] Model loaded')
  return session
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = dataUrl
  })
}

/**
 * Resize image to 512x512, optional contrast stretch, convert to CHW float32, ImageNet normalize.
 * If flipHorizontal, mirror the image for test-time augmentation.
 */
function preprocessImage(img: HTMLImageElement, flipHorizontal = false): ort.Tensor {
  const canvas = document.createElement('canvas')
  canvas.width = IMAGE_SIZE
  canvas.height = IMAGE_SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas 2d not available')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  if (flipHorizontal) {
    ctx.translate(IMAGE_SIZE, 0)
    ctx.scale(-1, 1)
  }
  ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE)
  const data = imageData.data
  const size = IMAGE_SIZE * IMAGE_SIZE

  let r0: number, g0: number, b0: number
  if (CONTRAST_STRETCH) {
    const samples: number[] = []
    for (let i = 0; i < size; i += 17) {
      samples.push(0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2])
    }
    samples.sort((a, b) => a - b)
    const low = samples[Math.floor(samples.length * 0.02)] ?? 0
    const high = samples[Math.floor(samples.length * 0.98)] ?? 255
    const span = Math.max(1, high - low)
    for (let i = 0; i < size; i++) {
      data[i * 4] = Math.max(0, Math.min(255, ((data[i * 4] - low) / span) * 255))
      data[i * 4 + 1] = Math.max(0, Math.min(255, ((data[i * 4 + 1] - low) / span) * 255))
      data[i * 4 + 2] = Math.max(0, Math.min(255, ((data[i * 4 + 2] - low) / span) * 255))
    }
  }

  const tensorData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE)
  for (let i = 0; i < size; i++) {
    r0 = (data[i * 4] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    g0 = (data[i * 4 + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    b0 = (data[i * 4 + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    if (USE_BGR) {
      tensorData[i] = b0
      tensorData[size + i] = g0
      tensorData[2 * size + i] = r0
    } else {
      tensorData[i] = r0
      tensorData[size + i] = g0
      tensorData[2 * size + i] = b0
    }
  }

  console.log(
    '[FloorDetection] Preprocess: [1,3,' +
      IMAGE_SIZE +
      ',' +
      IMAGE_SIZE +
      '], RGB, ImageNet norm, contrastStretch=',
    CONTRAST_STRETCH,
  )
  return new ort.Tensor('float32', tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE])
}

/**
 * Softmax over classes per pixel; returns probability for floor class. Recover more floor with threshold.
 */
function softmaxFloorProb(
  logits: Float32Array,
  numClasses: number,
  height: number,
  width: number,
  floorClass: number,
): Float32Array {
  const spatialSize = height * width
  const out = new Float32Array(spatialSize)
  for (let i = 0; i < spatialSize; i++) {
    let maxLogit = -Infinity
    for (let c = 0; c < numClasses; c++) {
      const v = logits[c * spatialSize + i]
      if (v > maxLogit) maxLogit = v
    }
    let sum = 0
    for (let c = 0; c < numClasses; c++) {
      sum += Math.exp(logits[c * spatialSize + i] - maxLogit)
    }
    out[i] = Math.exp(logits[floorClass * spatialSize + i] - maxLogit) / sum
  }
  return out
}

/**
 * Argmax over class dimension. Output shape [1, 150, 128, 128] -> class map 128*128.
 */
function argmaxClassDim(
  logits: Float32Array,
  numClasses: number,
  height: number,
  width: number,
): Uint8Array {
  const spatialSize = height * width
  const out = new Uint8Array(spatialSize)
  for (let i = 0; i < spatialSize; i++) {
    let maxVal = -Infinity
    let maxClass = 0
    for (let c = 0; c < numClasses; c++) {
      const v = logits[c * spatialSize + i]
      if (v > maxVal) {
        maxVal = v
        maxClass = c
      }
    }
    out[i] = maxClass
  }
  return out
}

/**
 * Morphological ops with configurable radius (1 = 3x3, 2 = 5x5).
 */
function morphologicalClose(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
  radius: number,
): Uint8ClampedArray {
  const r = Math.max(1, radius)
  const temp = new Uint8ClampedArray(mask.length)
  const out = new Uint8ClampedArray(mask.length)
  const dilate = (src: Uint8ClampedArray, dst: Uint8ClampedArray) => {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let max = 0
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            const ny = y + dy
            const nx = x + dx
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              const v = src[ny * width + nx]
              if (v > max) max = v
            }
          }
        }
        dst[y * width + x] = max
      }
    }
  }
  const erode = (src: Uint8ClampedArray, dst: Uint8ClampedArray) => {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let min = 255
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            const ny = y + dy
            const nx = x + dx
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              const v = src[ny * width + nx]
              if (v < min) min = v
            }
          }
        }
        dst[y * width + x] = min
      }
    }
  }
  dilate(mask, temp)
  erode(temp, out)
  return out
}

/** Single erosion (3×3): shrink mask to remove thin extensions under furniture legs. */
function erodeOnce(mask: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let min = 255
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const v = mask[ny * width + nx]
            if (v < min) min = v
          }
        }
      }
      out[y * width + x] = min
    }
  }
  return out
}

/** Single dilation (3×3) to recover thin edges. */
function dilateOnce(mask: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let max = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const v = mask[ny * width + nx]
            if (v > max) max = v
          }
        }
      }
      out[y * width + x] = max
    }
  }
  return out
}

/** Vertical-weighted dilation (kernel 1×5): expand floor mainly upward/downward to fill under rug top edges. */
function dilateVerticalOnce(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.length)
  const radiusY = 2
  const radiusX = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let max = 0
      for (let dy = -radiusY; dy <= radiusY; dy++) {
        for (let dx = -radiusX; dx <= radiusX; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const v = mask[ny * width + nx]
            if (v > max) max = v
          }
        }
      }
      out[y * width + x] = max
    }
  }
  return out
}

/**
 * Fill holes (enclosed background) inside floor mask. Holes = 0-regions not touching image boundary.
 */
function fillHoles(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
  maxHoleAreaRatio = 0.02,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask)
  const visited = new Uint8Array(mask.length)
  const totalPixels = mask.length
  const maxHoleArea = Math.floor(totalPixels * maxHoleAreaRatio)

  const stack: number[] = []
  const floodFromEdge = () => {
    for (let x = 0; x < width; x++) {
      const top = x
      const bottom = (height - 1) * width + x
      if (mask[top] === 0 && !visited[top]) {
        stack.push(top)
        visited[top] = 1
      }
      if (mask[bottom] === 0 && !visited[bottom]) {
        stack.push(bottom)
        visited[bottom] = 1
      }
    }
    for (let y = 0; y < height; y++) {
      const left = y * width
      const right = y * width + (width - 1)
      if (mask[left] === 0 && !visited[left]) {
        stack.push(left)
        visited[left] = 1
      }
      if (mask[right] === 0 && !visited[right]) {
        stack.push(right)
        visited[right] = 1
      }
    }
    while (stack.length > 0) {
      const i = stack.pop()!
      const y = (i / width) | 0
      const x = i % width
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const ni = ny * width + nx
            if (mask[ni] === 0 && !visited[ni]) {
              visited[ni] = 1
              stack.push(ni)
            }
          }
        }
      }
    }
  }
  floodFromEdge()

  for (let seed = 0; seed < mask.length; seed++) {
    if (mask[seed] !== 0 || visited[seed]) continue
    stack.length = 0
    stack.push(seed)
    const points: number[] = []
    visited[seed] = 1
    while (stack.length > 0) {
      const i = stack.pop()!
      points.push(i)
      const y = (i / width) | 0
      const x = i % width
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const ni = ny * width + nx
            if (mask[ni] === 0 && !visited[ni]) {
              visited[ni] = 1
              stack.push(ni)
            }
          }
        }
      }
    }
    if (points.length <= maxHoleArea) {
      for (const idx of points) out[idx] = 255
    }
  }
  return out
}

/**
 * 3×3 mode (majority) filter: removes isolated pixels at boundaries.
 */
function modeFilter3x3(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let count255 = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (mask[ny * width + nx] === 255) count255++
          }
        }
      }
      out[y * width + x] = count255 >= 5 ? 255 : 0
    }
  }
  return out
}

/**
 * Remove small connected components (below MIN_COMPONENT_AREA_RATIO of total floor area).
 */
function removeSmallComponents(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask)
  const visited = new Uint8Array(mask.length)
  let totalFloor = 0
  for (let i = 0; i < mask.length; i++) if (mask[i] === 255) totalFloor++
  const minArea = Math.max(1, Math.floor(totalFloor * MIN_COMPONENT_AREA_RATIO))

  const stack: number[] = []
  for (let seed = 0; seed < mask.length; seed++) {
    if (mask[seed] !== 255 || visited[seed]) continue
    let count = 0
    stack.length = 0
    stack.push(seed)
    visited[seed] = 1
    const points: number[] = []
    while (stack.length > 0) {
      const i = stack.pop()!
      points.push(i)
      count++
      const y = (i / width) | 0
      const x = i % width
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          const ny = y + dy
          const nx = x + dx
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const ni = ny * width + nx
            if (mask[ni] === 255 && !visited[ni]) {
              visited[ni] = 1
              stack.push(ni)
            }
          }
        }
      }
    }
    if (count < minArea) {
      for (let k = 0; k < points.length; k++) out[points[k]] = 0
    }
  }
  return out
}

/**
 * Resize mask to target size using nearest-neighbor (imageSmoothingEnabled = false).
 */
function resizeMask(
  maskSmall: Uint8ClampedArray,
  smallW: number,
  smallH: number,
  targetW: number,
  targetH: number,
): Uint8ClampedArray {
  const canvas = document.createElement('canvas')
  canvas.width = smallW
  canvas.height = smallH
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas 2d not available')
  const imageData = ctx.createImageData(smallW, smallH)
  for (let i = 0; i < maskSmall.length; i++) {
    const v = maskSmall[i]
    imageData.data[i * 4] = v
    imageData.data[i * 4 + 1] = v
    imageData.data[i * 4 + 2] = v
    imageData.data[i * 4 + 3] = 255
  }
  ctx.putImageData(imageData, 0, 0)

  const outCanvas = document.createElement('canvas')
  outCanvas.width = targetW
  outCanvas.height = targetH
  const outCtx = outCanvas.getContext('2d')
  if (!outCtx) throw new Error('Canvas 2d not available')
  outCtx.imageSmoothingEnabled = false
  outCtx.drawImage(canvas, 0, 0, targetW, targetH)
  const outData = outCtx.getImageData(0, 0, targetW, targetH)
  const out = new Uint8ClampedArray(targetW * targetH)
  for (let i = 0; i < out.length; i++) {
    out[i] = outData.data[i * 4] > 128 ? 255 : 0
  }
  return out
}

export interface FloorMaskResult {
  mask: Uint8ClampedArray
  width: number
  height: number
  /** PNG data URL for CSS mask-image (alpha = mask, 255 = opaque) */
  floorMaskDataUrl: string
  /** Red overlay data URL for debugging (mask area in red, alpha 0.5) */
  maskDebugDataUrl?: string
}

/** Gilam kesilishi yumshoq bo‘lishi uchun maskani blur qiladi (burchaklar yo‘qoladi). */
function smoothMaskForCarpet(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
  blurPx: number,
): Uint8ClampedArray {
  if (blurPx <= 0) return mask
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return mask
  const imageData = ctx.createImageData(width, height)
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i]
    imageData.data[i * 4] = 255
    imageData.data[i * 4 + 1] = 255
    imageData.data[i * 4 + 2] = 255
    imageData.data[i * 4 + 3] = v
  }
  ctx.putImageData(imageData, 0, 0)
  const out = document.createElement('canvas')
  out.width = width
  out.height = height
  const outCtx = out.getContext('2d')
  if (!outCtx) return mask
  outCtx.filter = `blur(${blurPx}px)`
  outCtx.drawImage(canvas, 0, 0)
  const blurred = outCtx.getImageData(0, 0, width, height)
  const smooth = new Uint8ClampedArray(mask.length)
  for (let i = 0; i < mask.length; i++) smooth[i] = blurred.data[i * 4 + 3]
  return smooth
}

/** Create PNG data URL for CSS mask-image: white with alpha = mask value. */
function maskToCssMaskDataUrl(mask: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return ''
  const imageData = ctx.createImageData(width, height)
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i]
    imageData.data[i * 4] = 255
    imageData.data[i * 4 + 1] = 255
    imageData.data[i * 4 + 2] = 255
    imageData.data[i * 4 + 3] = v
  }
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/png')
}

/** Flip mask horizontally (for TTA merge). */
function flipMaskHorizontal(
  mask: Uint8ClampedArray,
  width: number,
  height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.length)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      out[y * width + x] = mask[y * width + (width - 1 - x)]
    }
  }
  return out
}

/** Merge two masks with AND: only floor where both agree. */
function mergeMasksAnd(
  mask1: Uint8ClampedArray,
  mask2: Uint8ClampedArray,
  _width: number,
  _height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask1.length)
  for (let i = 0; i < out.length; i++) {
    out[i] = mask1[i] === 255 && mask2[i] === 255 ? 255 : 0
  }
  return out
}

/** Create a red overlay image from mask for visual debug (qirra yumshoqlamaydi). */
function maskToRedOverlayDataUrl(mask: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return ''
  const imageData = ctx.createImageData(width, height)
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i]
    imageData.data[i * 4] = 255
    imageData.data[i * 4 + 1] = 0
    imageData.data[i * 4 + 2] = 0
    imageData.data[i * 4 + 3] = v === 255 ? 200 : 0
  }
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL('image/png')
}

/** Run single inference and build raw floor mask (before post-processing). */
async function runSingleInference(
  img: HTMLImageElement,
  flipHorizontal: boolean,
): Promise<{ floorMaskSmall: Uint8ClampedArray; outW: number; outH: number }> {
  const inputTensor = preprocessImage(img, flipHorizontal)
  const session = await getSession()
  const feeds: Record<string, ort.Tensor> = {}
  feeds[session.inputNames[0]] = inputTensor

  const results = await session.run(feeds)
  const outputTensor = results[session.outputNames[0]]
  if (!outputTensor || !(outputTensor instanceof ort.Tensor)) {
    throw new Error('Model did not return a tensor')
  }

  const dims = outputTensor.dims
  const data = outputTensor.data as Float32Array
  const outH = dims.length >= 3 ? dims[dims.length - 2] : 128
  const outW = dims.length >= 3 ? dims[dims.length - 1] : 128
  const smallSize = outH * outW

  let floorMaskSmall: Uint8ClampedArray

  if (dims.length === 4 && dims[1] === 1) {
    floorMaskSmall = new Uint8ClampedArray(smallSize)
    for (let i = 0; i < smallSize; i++) {
      const v = data[i]
      floorMaskSmall[i] = v > BINARY_THRESHOLD ? 255 : 0
    }
  } else if (dims.length === 4 && dims[1] > 1) {
    const numClasses = dims[1]
    const classMap = argmaxClassDim(data, numClasses, outH, outW)

    const floorProb =
      USE_SOFTMAX_THRESHOLD || USE_HYBRID_AND
        ? softmaxFloorProb(data, numClasses, outH, outW, FLOOR_CLASS_INDEX)
        : null
    floorMaskSmall = new Uint8ClampedArray(smallSize)
    if (USE_HYBRID_AND && floorProb) {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] =
          classMap[i] === FLOOR_CLASS_INDEX && floorProb[i] >= FLOOR_PROB_THRESHOLD ? 255 : 0
      }
    } else if (USE_SOFTMAX_THRESHOLD && floorProb) {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] = floorProb[i] >= FLOOR_PROB_THRESHOLD ? 255 : 0
      }
    } else {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] = classMap[i] === FLOOR_CLASS_INDEX ? 255 : 0
      }
    }

    if (SUBTRACT_FURNITURE_FROM_FLOOR) {
      for (let i = 0; i < smallSize; i++) {
        if (FURNITURE_CLASS_INDICES.has(classMap[i])) floorMaskSmall[i] = 0
      }
    }

  } else {
    throw new Error(`Unexpected output shape: ${JSON.stringify(dims)}`)
  }

  return { floorMaskSmall, outW, outH }
}

export async function detectFloorWithSegFormer(imageDataUrl: string): Promise<FloorMaskResult> {
  const img = await loadImage(imageDataUrl)
  const width = img.naturalWidth
  const height = img.naturalHeight
  console.log('[FloorDetection] Original image size', width, 'x', height)

  let floorMaskSmall: Uint8ClampedArray
  let outW: number
  let outH: number

  if (USE_TTA) {
    console.log('[FloorDetection] Running TTA (original + flipped)...')
    const [r1, r2] = await Promise.all([
      runSingleInference(img, false),
      runSingleInference(img, true),
    ])
    outW = r1.outW
    outH = r1.outH
    const mask2Flipped = flipMaskHorizontal(r2.floorMaskSmall, outW, outH)
    floorMaskSmall = mergeMasksAnd(r1.floorMaskSmall, mask2Flipped, outW, outH)
    console.log('[FloorDetection] TTA merge complete')
  } else {
    console.log('[FloorDetection] Running inference...')
    const r = await runSingleInference(img, false)
    floorMaskSmall = r.floorMaskSmall
    outW = r.outW
    outH = r.outH
  }

  // Post-process: close, remove speckle, erosion to exclude legs, opening (dilate), mode filter, hole fill
  floorMaskSmall = morphologicalClose(floorMaskSmall, outW, outH, MORPH_CLOSE_RADIUS)
  floorMaskSmall = removeSmallComponents(floorMaskSmall, outW, outH)
  for (let k = 0; k < ERODE_ITERATIONS; k++) {
    floorMaskSmall = erodeOnce(floorMaskSmall, outW, outH)
  }
  if (ERODE_ITERATIONS > 0) {
    floorMaskSmall = removeSmallComponents(floorMaskSmall, outW, outH)
  }
  for (let k = 0; k < EDGE_DILATE_ITERATIONS; k++) {
    floorMaskSmall = dilateOnce(floorMaskSmall, outW, outH)
  }
  for (let k = 0; k < VERTICAL_DILATE_ITERATIONS; k++) {
    floorMaskSmall = dilateVerticalOnce(floorMaskSmall, outW, outH)
  }

  floorMaskSmall = modeFilter3x3(floorMaskSmall, outW, outH)
  floorMaskSmall = fillHoles(floorMaskSmall, outW, outH)

  // Resize to original image size with nearest-neighbor
  const mask = resizeMask(floorMaskSmall, outW, outH, width, height)
  console.log('[FloorDetection] Mask post-processed and resized to', width, 'x', height)

  // Gilam kesilishi burchaksiz (yumshoq) bo‘lishi uchun maskani yengil blur qilamiz
  const blurPx = Math.max(2, Math.min(12, Math.round(width / 200)))
  const maskForCarpet = smoothMaskForCarpet(mask, width, height, blurPx)
  const floorMaskDataUrl = maskToCssMaskDataUrl(maskForCarpet, width, height)
  const maskDebugDataUrl = maskToRedOverlayDataUrl(mask, width, height)

  return { mask, width, height, floorMaskDataUrl, maskDebugDataUrl }
}
