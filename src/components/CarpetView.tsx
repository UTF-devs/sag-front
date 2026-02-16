"use client";

import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { RefreshCw, Image as ImageIcon, ArrowUpDown } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";
import { detectFloorWithSegFormer } from "../utils/floorDetection";
import type { FloorMaskResult } from "../utils/floorDetection";
import type { Carpet } from "../types/carpet";

export type CarpetTransform = {
  position: { x: number; y: number };
  size: { width: number; height: number };
  rotation: number;
};

type CarpetViewProps = {
  carpet: Carpet;
  onChangeCarpet: () => void;
  initialTransform?: Partial<CarpetTransform>;
};

const DEFAULT_TRANSFORM: CarpetTransform = {
  position: { x: 50, y: 60 },
  size: { width: 40, height: 30 },
  rotation: 0,
};

function calculateAngle(
  centerX: number,
  centerY: number,
  pointX: number,
  pointY: number,
) {
  const dx = pointX - centerX;
  const dy = pointY - centerY;
  return Math.atan2(dy, dx) * (180 / Math.PI);
}

export default function CarpetView({
  carpet,
  onChangeCarpet,
  initialTransform,
}: CarpetViewProps) {
  const { t } = useLanguage();
  const STORAGE_KEY = `carpet_view_${carpet.id}`;

  // Load from localStorage on mount
  const [roomImage, setRoomImage] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        return data.roomImage || null;
      }
    }
    return null;
  });

  console.log("roomImage", carpet);

  const [floorResult, setFloorResult] = useState<FloorMaskResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const roomChangeRef = useRef<HTMLInputElement>(null);

  const [carpetPosition, setCarpetPosition] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        return (
          data.carpetPosition ||
          (initialTransform?.position ?? DEFAULT_TRANSFORM.position)
        );
      }
    }
    return initialTransform?.position ?? DEFAULT_TRANSFORM.position;
  });

  const [carpetSize, setCarpetSize] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        return (
          data.carpetSize || (initialTransform?.size ?? DEFAULT_TRANSFORM.size)
        );
      }
    }
    return initialTransform?.size ?? DEFAULT_TRANSFORM.size;
  });

  const [carpetRotation, setCarpetRotation] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved);
        return (
          data.carpetRotation ??
          initialTransform?.rotation ??
          DEFAULT_TRANSFORM.rotation
        );
      }
    }
    return initialTransform?.rotation ?? DEFAULT_TRANSFORM.rotation;
  });

  const [isDragging, setIsDragging] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isSlidingSize, setIsSlidingSize] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [rotationStart, setRotationStart] = useState(0);
  const sizeSliderTrackRef = useRef<HTMLDivElement>(null);
  const rotationCenterRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const previewContainerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });

  useEffect(() => {
    const el = previewContainerRef.current;
    if (!el) return;
    const update = () =>
      setContainerSize({ w: el.clientWidth, h: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [roomImage, floorResult]);

  // localStorage'dan xona yuklanganida yoki sahifa yangilanganda polni qayta aniqlash (floorResult saqlanmaydi)
  useEffect(() => {
    if (!roomImage || floorResult !== null || isLoading) return;
    let cancelled = false;
    setIsLoading(true);
    detectFloorWithSegFormer(roomImage)
      .then((result) => {
        if (!cancelled) setFloorResult(result);
      })
      .catch((err) => {
        if (!cancelled)
          console.error("[CarpetView] Floor detection (restore) error", err);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [roomImage]);

  // Maskani qizil overlay (object-cover) bilan ustma-ust qilish; useMemo — qayta hisoblashni kamaytiradi
  const maskStyle = useMemo(() => {
    if (!floorResult?.floorMaskDataUrl) {
      console.log("[CarpetView] No mask style - floorResult:", floorResult);
      return undefined;
    }
    const hasMaskAlign =
      containerSize.w > 0 &&
      containerSize.h > 0 &&
      floorResult.width > 0 &&
      floorResult.height > 0;
    if (hasMaskAlign) {
      const scale = Math.max(
        containerSize.w / floorResult.width,
        containerSize.h / floorResult.height,
      );
      const w = Math.ceil(floorResult.width * scale);
      const h = Math.ceil(floorResult.height * scale);
      return {
        WebkitMaskImage: `url(${floorResult.floorMaskDataUrl})`,
        maskImage: `url(${floorResult.floorMaskDataUrl})`,
        maskSize: `${w}px ${h}px`,
        maskRepeat: "no-repeat" as const,
        maskPosition: "center" as const,
      };
    }
    return {
      WebkitMaskImage: `url(${floorResult.floorMaskDataUrl})`,
      maskImage: `url(${floorResult.floorMaskDataUrl})`,
      maskSize: "100% 100%",
      maskRepeat: "no-repeat" as const,
      maskPosition: "0 0" as const,
    };
  }, [
    floorResult?.floorMaskDataUrl,
    floorResult?.width,
    floorResult?.height,
    containerSize.w,
    containerSize.h,
  ]);

  // Save to localStorage whenever state changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      const dataToSave = {
        roomImage,
        carpetPosition,
        carpetSize,
        carpetRotation,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(dataToSave));
    }
  }, [roomImage, carpetPosition, carpetSize, carpetRotation, STORAGE_KEY]);

  const updateSizeFromSlider = useCallback((clientY: number) => {
    const track = sizeSliderTrackRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const raw = ((rect.bottom - clientY) / rect.height) * 100;
    const scale = Math.max(0, Math.min(100, raw));
    const width = 15 + ((80 - 15) * scale) / 100;
    const height = 15 + ((60 - 15) * scale) / 100;
    setCarpetSize({ width, height });
  }, []);

  // Gilam va aylantirish tugmasi qatlami uchun bir xil joy/transform (DRY)
  const carpetContainerStyle = useMemo(
    (): React.CSSProperties => ({
      left: `${carpetPosition.x}%`,
      top: `${carpetPosition.y}%`,
      width: `${carpetSize.width}%`,
      height: `${carpetSize.height}%`,
      transform: `translate(-50%, -50%) perspective(1000px) rotateX(74deg) rotateZ(${carpetRotation}deg)`,
      transformStyle: "preserve-3d",
      transition: isRotating ? "none" : "transform 0.1s ease-out",
      willChange: isRotating ? "transform" : "auto",
    }),
    [
      carpetPosition.x,
      carpetPosition.y,
      carpetSize.width,
      carpetSize.height,
      carpetRotation,
      isRotating,
    ],
  );

  // Get image URL - use proxy only in development, direct URL in production
  const getImageUrl = useCallback((url: string | undefined): string => {
    if (!url) return "";
    
    // If already proxied, extract direct URL first
    if (url.includes("/api/carpet-image?url=")) {
      try {
        const decoded = decodeURIComponent(url.split("url=")[1] || url);
        url = decoded;
      } catch {
        // Keep original if decode fails
      }
    }
    
    // Check if we're in production (proxy endpoint may not exist)
    const isProduction = window.location.hostname !== 'localhost' && 
                         window.location.hostname !== '127.0.0.1';
    
    // In production, use direct URL (backend should handle CORS)
    // In development, use proxy to avoid CORS issues
    if (!isProduction && (url.startsWith("http://") || url.startsWith("https://"))) {
      try {
        const parsed = new URL(url);
        // Only proxy if it's a different origin
        if (parsed.origin !== window.location.origin) {
          const proxiedUrl = `/api/carpet-image?url=${encodeURIComponent(url)}`;
          console.log("[CarpetView] Using proxy URL for CORS (dev):", proxiedUrl);
          return proxiedUrl;
        }
      } catch {
        // If URL parsing fails, return as is
      }
    }
    
    // Production: use direct URL
    console.log("[CarpetView] Using direct URL:", url);
    return url;
  }, []);

  const imageUrl = useMemo(() => getImageUrl(carpet?.image), [carpet?.image, getImageUrl]);

  // Debug: carpet image URL'ni console'ga chiqarish (carpetContainerStyle va maskStyle dan keyin)
  useEffect(() => {
    if (carpet?.image) {
      console.log("[CarpetView] Carpet image URL (original):", carpet.image);
      console.log("[CarpetView] Carpet image URL (final):", imageUrl);
      console.log("[CarpetView] Carpet container style:", carpetContainerStyle);
      console.log("[CarpetView] Mask style:", maskStyle);
    }
  }, [carpet?.image, imageUrl, carpetContainerStyle, maskStyle]);

  // Slider tortilganda document bo‘yicha move/up qabul qilish
  useEffect(() => {
    if (!isSlidingSize) return;
    const onMove = (e: MouseEvent) => updateSizeFromSlider(e.clientY);
    const onUp = () => setIsSlidingSize(false);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) updateSizeFromSlider(e.touches[0].clientY);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.addEventListener("touchmove", onTouchMove, { passive: false });
    document.addEventListener("touchend", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.removeEventListener("touchmove", onTouchMove);
      document.removeEventListener("touchend", onUp);
    };
  }, [isSlidingSize, updateSizeFromSlider]);

  const handleRoomUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    const reader = new FileReader();
    reader.onload = async (event) => {
      const dataUrl = event.target?.result;
      if (typeof dataUrl !== "string") return;
      setFloorResult(null);
      setIsLoading(true);
      setRoomImage(dataUrl);
      try {
        const result = await detectFloorWithSegFormer(dataUrl);
        setFloorResult(result);
      } catch (err) {
        console.error("[CarpetView] Floor detection error", err);
        setFloorResult(null);
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleChangeRoom = () => {
    roomChangeRef.current?.click();
  };

  // Hozirgi o'lchamdan slider pozitsiyasi 0–100 (yuqori = katta)
  const sizeScalePercent = ((carpetSize.width - 15) / (80 - 15)) * 100;

  const handleSizeSliderMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsSlidingSize(true);
    updateSizeFromSlider(e.clientY);
  };

  const handleSizeSliderTouchStart = (e: React.TouchEvent) => {
    e.stopPropagation();
    setIsSlidingSize(true);
    updateSizeFromSlider(e.touches[0].clientY);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleRotationMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsRotating(true);
    const carpetEl = document.querySelector(
      "[data-carpet-container]",
    ) as HTMLElement;
    if (carpetEl) {
      const rect = carpetEl.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      rotationCenterRef.current = { x: centerX, y: centerY };
      const angle = calculateAngle(centerX, centerY, e.clientX, e.clientY);
      setRotationStart(angle - carpetRotation);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -5 : 5;
    setCarpetRotation((prev: number) => (prev + delta) % 360);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isRotating) {
      e.preventDefault();
      const { x: centerX, y: centerY } = rotationCenterRef.current;
      const currentAngle = calculateAngle(
        centerX,
        centerY,
        e.clientX,
        e.clientY,
      );
      const newRotation = currentAngle - rotationStart;
      setCarpetRotation(newRotation);
    } else if (isSlidingSize) {
      e.preventDefault();
      updateSizeFromSlider(e.clientY);
    } else if (isDragging) {
      e.preventDefault();
      const deltaX = (e.clientX - dragStart.x) / 5;
      const deltaY = (e.clientY - dragStart.y) / 5;
      setCarpetPosition((prev: { x: number; y: number }) => ({
        x: Math.max(10, Math.min(90, prev.x + deltaX)),
        y: Math.max(10, Math.min(90, prev.y + deltaY)),
      }));
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsRotating(false);
    setIsSlidingSize(false);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    const touch = e.touches[0];
    setIsDragging(true);
    setDragStart({ x: touch.clientX, y: touch.clientY });
  };

  const handleRotationTouchStart = (e: React.TouchEvent) => {
    e.stopPropagation();
    setIsRotating(true);
    const touch = e.touches[0];
    const carpetEl = document.querySelector(
      "[data-carpet-container]",
    ) as HTMLElement;
    if (carpetEl) {
      const rect = carpetEl.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      rotationCenterRef.current = { x: centerX, y: centerY };
      const angle = calculateAngle(
        centerX,
        centerY,
        touch.clientX,
        touch.clientY,
      );
      setRotationStart(angle - carpetRotation);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isRotating) {
      e.preventDefault();
      const touch = e.touches[0];
      const { x: centerX, y: centerY } = rotationCenterRef.current;
      const currentAngle = calculateAngle(
        centerX,
        centerY,
        touch.clientX,
        touch.clientY,
      );
      const newRotation = currentAngle - rotationStart;
      setCarpetRotation(newRotation);
    } else if (isSlidingSize) {
      e.preventDefault();
      updateSizeFromSlider(e.touches[0].clientY);
    } else if (isDragging) {
      e.preventDefault();
      const touch = e.touches[0];
      const deltaX = (touch.clientX - dragStart.x) / 5;
      const deltaY = (touch.clientY - dragStart.y) / 5;
      setCarpetPosition((prev: { x: number; y: number }) => ({
        x: Math.max(10, Math.min(90, prev.x + deltaX)),
        y: Math.max(10, Math.min(90, prev.y + deltaY)),
      }));
      setDragStart({ x: touch.clientX, y: touch.clientY });
    }
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
    setIsRotating(false);
    setIsSlidingSize(false);
  };

  return (
    <div
      className=" flex w-full gap-8"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      <div className="max-w-4xl w-full mx-auto h-[900px]">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleRoomUpload}
          accept="image/*"
          className="hidden"
        />

        {!roomImage ? (
          <div className=" rounded-lg p-12 text-center">
            <div className="flex flex-col items-center">
              <ImageIcon className="text-gray-400 mb-4" size={64} />
              <h3 className="text-black text-2xl font-semibold mb-4">
                {t("carpet_view.upload_title")}
              </h3>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-[#CCA57A] text-white px-8 py-3 rounded-lg font-semibold hover:bg-main_color/90 transition"
              >
                {t("carpet_view.choose_image")}
              </button>
            </div>
          </div>
        ) : isLoading ? (
          <div className="bg-gray-800 rounded-lg overflow-hidden mb-6 relative">
            <img
              src={roomImage}
              alt="Your room"
              className="opacity-50 w-full h-[500px] object-cover"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="flex flex-col items-center">
                <RefreshCw
                  className="animate-spin text-blue-400 mb-4"
                  size={48}
                />
                <p className="text-white text-xl">{t("carpet_view.loading")}</p>
              </div>
            </div>
          </div>
        ) : (
          <div>
            <div
              ref={previewContainerRef}
              className="bg-gray-800 rounded-lg overflow-hidden mb-6 relative select-none"
            >
              <img
                src={roomImage}
                alt="Your room"
                className="w-full h-[500px] object-cover"
                style={{ objectFit: "cover" }}
                draggable={false}
              />
              {/* red floor */}
              {/* {floorResult?.maskDebugDataUrl && (
                <div
                  className='absolute inset-0 w-full h-full pointer-events-none'
                  style={{ top: 0, left: 0, right: 0, bottom: 0 }}
                  aria-hidden
                >
                  <img
                    src={floorResult.maskDebugDataUrl}
                    alt=''
                    className='w-full h-full object-cover object-center'
                    style={{ opacity: 0.9 }}
                  />
                </div>
              )} */}
              <div
                className="absolute left-0 right-0 pointer-events-none z-[3]"
                style={{
                  top: 10,
                  bottom: 0,
                  ...maskStyle,
                  visibility: "visible",
                }}
              >
                <div
                  data-carpet-container
                  className="absolute touch-none pointer-events-auto"
                  style={{
                    ...carpetContainerStyle,
                    cursor: isDragging ? "grabbing" : "grab",
                    zIndex: 3,
                    visibility: "visible",
                  }}
                  onMouseDown={handleMouseDown}
                  onTouchStart={handleTouchStart}
                  onWheel={handleWheel}
                >
                  <div className="relative w-full h-full" style={{ zIndex: 3 }}>
                    <img
                      src={imageUrl}
                      alt={carpet?.name || "Carpet"}
                      className="shadow-2xl pointer-events-none w-full h-full"
                      style={{
                        opacity: 1,
                        display: "block",
                        visibility: "visible",
                        zIndex: 3,
                        position: "relative",
                        // objectFit: "cover",
                        width: "100%",
                        height: "100%",
                      }}
                      draggable={false}
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        console.error(
                          "[CarpetView] Carpet image failed to load:",
                          {
                            originalUrl: carpet?.image,
                            finalUrl: imageUrl,
                            currentSrc: target.src,
                            isProxyUrl: target.src.includes("/api/carpet-image"),
                            error: "CORS or network error - check proxy endpoint"
                          },
                        );
                      }}
                      onLoad={(e) => {
                        const target = e.target as HTMLImageElement;
                        console.log(
                          "[CarpetView] Carpet image loaded successfully:",
                          {
                            originalUrl: carpet?.image,
                            finalUrl: imageUrl,
                            loadedSrc: target.src,
                            dimensions: `${target.naturalWidth}x${target.naturalHeight}`,
                          },
                        );
                      }}
                    />
                  </div>
                </div>
              </div>
              {/* Aylantirish tugmasi maskdan tashqarida — pol kesmasa ham har doim ko‘rinadi; gilam bilan bir xil transform */}
              <div
                className="absolute left-0 right-0 pointer-events-none"
                style={{ top: 20, bottom: 0 }}
              >
                <div
                  className="absolute touch-none pointer-events-none"
                  style={carpetContainerStyle}
                >
                  <div
                    className="absolute -top-12 right-10 w-10 h-10 bg-green-500 rounded-full hover:bg-green-600 flex items-center justify-center text-white shadow-xl transition-all hover:scale-110 pointer-events-auto"
                    onMouseDown={handleRotationMouseDown}
                    onTouchStart={handleRotationTouchStart}
                    style={{ cursor: isRotating ? "grabbing" : "grab" }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
                    </svg>
                  </div>
                </div>
              </div>
              <div
                ref={sizeSliderTrackRef}
                className="absolute left-4 top-1/2 -translate-y-1/2 w-2 h-40 bg-gray-600 rounded-full flex flex-col justify-end pointer-events-auto"
                aria-label="O'lcham"
              >
                <div
                  className="absolute left-1/2 w-6 h-6 -translate-x-1/2 -translate-y-1/2 rounded-full bg-green-500 hover:bg-green-600 cursor-ns-resize shadow-lg flex items-center justify-center pointer-events-auto touch-none"
                  style={{
                    top: `${100 - Math.max(0, Math.min(100, sizeScalePercent))}%`,
                  }}
                  onMouseDown={handleSizeSliderMouseDown}
                  onTouchStart={handleSizeSliderTouchStart}
                  role="slider"
                  aria-valuenow={Math.round(sizeScalePercent)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <ArrowUpDown className="text-white" size={14} />
                </div>
              </div>
            </div>

            <div className="flex gap-4">
              <input
                type="file"
                ref={roomChangeRef}
                onChange={handleRoomUpload}
                accept="image/*"
                className="hidden"
              />
              <button
                onClick={handleChangeRoom}
                className="flex-1 bg-gray-700 text-white py-3 rounded-lg font-semibold hover:bg-gray-600 transition flex items-center justify-center"
              >
                <ImageIcon className="mr-2" size={20} />
                {t("carpet_view.change_room")}
              </button>
              <button
                onClick={onChangeCarpet}
                className="flex-1 bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition flex items-center justify-center"
              >
                <RefreshCw className="mr-2" size={20} />
                {t("carpet_view.change_carpet")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
