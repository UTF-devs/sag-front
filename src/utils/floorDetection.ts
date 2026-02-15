import * as ort from 'onnxruntime-web'

/** Model input size. Use 512 unless your ONNX model supports dynamic shape (e.g. 768/1024 for finer rug/floor edges). */
const IMAGE_SIZE = 512
const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]
const USE_BGR = false
const FLOOR_CLASS_INDEX = 3
const BINARY_THRESHOLD = 0.5
/** Keep small floor patches (e.g. under rugs). 0.003 = 0.3% of floor area; only remove speckle. */
const MIN_COMPONENT_AREA_RATIO = 0.003
/** true = use softmax threshold; false = argmax only. */
const USE_SOFTMAX_THRESHOLD = true
/** Include pixel as floor if P(floor) >= this. Yuqoriroq (0.32–0.35) = pol tepaga chiqmaydi, devor/mebel kirmaydi. */
const FLOOR_PROB_THRESHOLD = 0.34
/** Combine argmax + softmax: floor if (argmax=floor) OR (P(floor)>=threshold). Preserves boundaries + floor under carpets. */
const USE_HYBRID_ARGMAX_SOFTMAX = true
const CONTRAST_STRETCH = true
/** 3×3 close: kichik teshiklarni yopadi, chegarani haddan ortiq kengaytirmaydi. 2 = 5×5 (teparoqga chiqib ketadi). */
const MORPH_CLOSE_RADIUS = 1
/** Isotropic dilation: 0 = pol kengaymasin, qizil tepaga chiqmasin. */
const EDGE_DILATE_ITERATIONS = 0
/** Vertical dilation o‘chirildi: pol yuqoriga kengayib gilam tepasini kesmasin, devor/mebelga chiqmasin. */
const VERTICAL_DILATE_ITERATIONS = 0

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
 * Returns tensor shape [1, 3, 512, 512].
 */
function preprocessImage(img: HTMLImageElement): ort.Tensor {
  const canvas = document.createElement('canvas')
  canvas.width = IMAGE_SIZE
  canvas.height = IMAGE_SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas 2d not available')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
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

export async function detectFloorWithSegFormer(imageDataUrl: string): Promise<FloorMaskResult> {
  const img = await loadImage(imageDataUrl)
  const width = img.naturalWidth
  const height = img.naturalHeight
  console.log('[FloorDetection] Original image size', width, 'x', height)

  const inputTensor = preprocessImage(img)
  const session = await getSession()

  const feeds: Record<string, ort.Tensor> = {}
  feeds[session.inputNames[0]] = inputTensor

  console.log('[FloorDetection] Running inference...')
  const results = await session.run(feeds)
  const outputName = session.outputNames[0]
  const outputTensor = results[outputName]
  if (!outputTensor || !(outputTensor instanceof ort.Tensor)) {
    throw new Error('Model did not return a tensor')
  }

  const dims = outputTensor.dims
  const data = outputTensor.data as Float32Array

  console.log('[FloorDetection] Output dims:', dims)
  console.log('[FloorDetection] Output data (first 50):', Array.from(data.slice(0, 50)))

  const outH = dims.length >= 3 ? dims[dims.length - 2] : 128
  const outW = dims.length >= 3 ? dims[dims.length - 1] : 128
  const smallSize = outH * outW

  let floorMaskSmall: Uint8ClampedArray

  if (dims.length === 4 && dims[1] === 1) {
    // Binary mask [1, 1, H, W] — tune BINARY_THRESHOLD (0.3–0.6) for best accuracy
    console.log('[FloorDetection] Binary mask, threshold:', BINARY_THRESHOLD)
    floorMaskSmall = new Uint8ClampedArray(smallSize)
    for (let i = 0; i < smallSize; i++) {
      const v = data[i]
      floorMaskSmall[i] = v > BINARY_THRESHOLD ? 255 : 0
    }
  } else if (dims.length === 4 && dims[1] > 1) {
    // Multi-class [1, C, H, W]
    const numClasses = dims[1]
    const classMap = argmaxClassDim(data, numClasses, outH, outW)
    const uniqueIds = [...new Set(classMap)].sort((a, b) => a - b)
    console.log('[FloorDetection] Unique class IDs:', uniqueIds)

    const floorProb =
      USE_SOFTMAX_THRESHOLD || USE_HYBRID_ARGMAX_SOFTMAX
        ? softmaxFloorProb(data, numClasses, outH, outW, FLOOR_CLASS_INDEX)
        : null
    floorMaskSmall = new Uint8ClampedArray(smallSize)
    if (USE_HYBRID_ARGMAX_SOFTMAX && floorProb) {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] =
          classMap[i] === FLOOR_CLASS_INDEX || floorProb[i] >= FLOOR_PROB_THRESHOLD ? 255 : 0
      }
      console.log('[FloorDetection] Hybrid: argmax=floor OR P(floor) >=', FLOOR_PROB_THRESHOLD)
    } else if (USE_SOFTMAX_THRESHOLD && floorProb) {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] = floorProb[i] >= FLOOR_PROB_THRESHOLD ? 255 : 0
      }
      console.log('[FloorDetection] Softmax threshold: P(floor) >=', FLOOR_PROB_THRESHOLD)
    } else {
      for (let i = 0; i < smallSize; i++) {
        floorMaskSmall[i] = classMap[i] === FLOOR_CLASS_INDEX ? 255 : 0
      }
    }

    const countByClass: Record<number, number> = {}
    for (let i = 0; i < classMap.length; i++) {
      const c = classMap[i]
      countByClass[c] = (countByClass[c] ?? 0) + 1
    }
    const sortedClasses = Object.entries(countByClass)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
    console.log(
      '[FloorDetection] Pixel count per class (top 15):',
      sortedClasses.map(([c, n]) => `class ${c}=${n}`).join(', '),
    )

    const floorPixels = floorMaskSmall.filter(v => v === 255).length
    console.log(
      '[FloorDetection] Floor class:',
      FLOOR_CLASS_INDEX,
      '| floor pixels:',
      floorPixels,
      '| total:',
      smallSize,
    )
  } else {
    console.error('[FloorDetection] Unexpected output shape:', dims)
    throw new Error(`Unexpected output shape: ${JSON.stringify(dims)}`)
  }

  // Post-process: 3×3 close, speckle olib tashlash, faqat isotropic dilate (vertical yo‘q — tepaga chiqmasin)
  floorMaskSmall = morphologicalClose(floorMaskSmall, outW, outH, MORPH_CLOSE_RADIUS)
  floorMaskSmall = removeSmallComponents(floorMaskSmall, outW, outH)
  for (let k = 0; k < EDGE_DILATE_ITERATIONS; k++) {
    floorMaskSmall = dilateOnce(floorMaskSmall, outW, outH)
  }
  for (let k = 0; k < VERTICAL_DILATE_ITERATIONS; k++) {
    floorMaskSmall = dilateVerticalOnce(floorMaskSmall, outW, outH)
  }

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
