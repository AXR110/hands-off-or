"use client";

import { useLayoutEffect, useRef, useState, type RefObject } from "react";

import {
  appendHandStroke,
  clearImageStrokes,
  paintStrokes,
  pointerToCanvas,
  strokesForImage,
  type DrawHand,
  type StrokeMap,
} from "./draw-strokes";

export type HandPointer = { x: number; y: number; pinch: boolean };

type ImageViewerProps = {
  imageId: string;
  src: string;
  alt: string;
  scale: number;
  pan: { x: number; y: number };
  imageRef: RefObject<HTMLImageElement | null>;
  pointer: HandPointer | null;
  secondaryPointer: HandPointer | null;
  highlighted: boolean;
  clearHighlighted: boolean;
};

function pointerOverFilmControl(x: number, y: number) {
  const hit = document.elementFromPoint(x * window.innerWidth, y * window.innerHeight);
  return Boolean(hit?.closest("[data-draw-mode], [data-clear-drawing]"));
}

function placeCanvasOverImage(image: HTMLImageElement, canvas: HTMLCanvasElement) {
  const origin =
    image.offsetParent instanceof HTMLElement ? image.offsetParent : image.parentElement;
  if (!origin) return false;
  const imageRect = image.getBoundingClientRect();
  const originRect = origin.getBoundingClientRect();
  if (imageRect.width < 1 || imageRect.height < 1) return false;
  canvas.style.left = `${imageRect.left - originRect.left}px`;
  canvas.style.top = `${imageRect.top - originRect.top}px`;
  canvas.style.width = `${imageRect.width}px`;
  canvas.style.height = `${imageRect.height}px`;
  const width = Math.max(1, Math.round(imageRect.width));
  const height = Math.max(1, Math.round(imageRect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return true;
}

export default function ImageViewer({
  imageId,
  src,
  alt,
  scale,
  pan,
  imageRef,
  pointer,
  secondaryPointer,
  highlighted,
  clearHighlighted,
}: ImageViewerProps) {
  const [isDrawMode, setIsDrawMode] = useState(false);
  const [paintVersion, setPaintVersion] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingsRef = useRef<StrokeMap>({});
  const drawingRef = useRef<Record<DrawHand, boolean>>({ primary: false, secondary: false });
  const activeImageRef = useRef(imageId);

  const clearDrawing = () => {
    drawingsRef.current = clearImageStrokes(drawingsRef.current, imageId);
    drawingRef.current = { primary: false, secondary: false };
    setPaintVersion((version) => version + 1);
  };

  useLayoutEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas) return;

    if (activeImageRef.current !== imageId) {
      activeImageRef.current = imageId;
      drawingRef.current = { primary: false, secondary: false };
    }

    const paintCurrent = () => {
      if (!placeCanvasOverImage(image, canvas)) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      paintStrokes(ctx, canvas.width, canvas.height, strokesForImage(drawingsRef.current, imageId));
    };

    if (isDrawMode && placeCanvasOverImage(image, canvas)) {
      const rect = canvas.getBoundingClientRect();
      const viewport = { width: window.innerWidth, height: window.innerHeight };
      let drawings = drawingsRef.current;

      const applyHand = (hand: HandPointer | null, key: DrawHand) => {
        if (!hand?.pinch || pointerOverFilmControl(hand.x, hand.y)) {
          drawingRef.current[key] = false;
          return;
        }
        const point = pointerToCanvas(hand.x, hand.y, rect, viewport);
        if (!point) {
          drawingRef.current[key] = false;
          return;
        }
        drawings = appendHandStroke(drawings, imageId, key, point, drawingRef.current[key]);
        drawingRef.current[key] = true;
      };

      applyHand(pointer, "primary");
      applyHand(secondaryPointer, "secondary");
      drawingsRef.current = drawings;
    } else {
      drawingRef.current = { primary: false, secondary: false };
    }

    paintCurrent();
    const observer = new ResizeObserver(() => paintCurrent());
    observer.observe(image);
    window.addEventListener("resize", paintCurrent);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", paintCurrent);
    };
  }, [imageId, imageRef, isDrawMode, paintVersion, pan.x, pan.y, pointer, scale, secondaryPointer]);

  return (
    <>
      <div className="absolute top-3 left-1/2 z-20 flex -translate-x-1/2 gap-2">
        <button
          type="button"
          data-draw-mode
          aria-pressed={isDrawMode}
          onClick={() => setIsDrawMode((current) => !current)}
          className={`min-h-14 rounded-lg border px-5 py-3 text-sm font-semibold tracking-[0.18em] uppercase transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-400 ${
            isDrawMode
              ? "border-red-700 bg-red-600 text-white shadow-[0_0_28px_rgba(220,38,38,0.75)]"
              : highlighted
                ? "border-sky-200 bg-sky-400 text-slate-950 shadow-[0_0_24px_rgba(56,189,248,0.6)]"
                : "border-slate-600 bg-slate-950/80 text-slate-200 hover:border-slate-400"
          }`}
        >
          Draw mode
        </button>
        <button
          type="button"
          data-clear-drawing
          onClick={clearDrawing}
          className={`min-h-14 rounded-lg border px-5 py-3 text-sm font-semibold tracking-[0.18em] uppercase transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-300 ${
            clearHighlighted
              ? "border-sky-200 bg-sky-400 text-slate-950 shadow-[0_0_24px_rgba(56,189,248,0.6)]"
              : "border-slate-600 bg-slate-950/80 text-slate-200 hover:border-slate-400"
          }`}
        >
          Clear drawing
        </button>
      </div>
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        draggable={false}
        className="max-h-full max-w-full origin-center object-contain select-none"
        style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})` }}
      />
      <canvas
        ref={canvasRef}
        data-draw-canvas
        data-draw-image={imageId}
        className="pointer-events-none absolute z-10"
      />
    </>
  );
}
