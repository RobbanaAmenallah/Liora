import React, { useEffect, useRef, useState } from 'react';
import { DataPoint, NormalizationStats } from '@/types/ml';
import { Eye, EyeOff, Crosshair } from 'lucide-react';

interface DecisionBoundaryCanvasProps {
  points: DataPoint[];
  bounds: { minX1: number; maxX1: number; minX2: number; maxX2: number };
  stats: NormalizationStats;
  gridPredictions: Float32Array;
  gridResolution: number;
  presentationMode?: boolean;
}

const DecisionBoundaryCanvasInner: React.FC<DecisionBoundaryCanvasProps> = ({
  points,
  bounds,
  gridPredictions,
  gridResolution,
  presentationMode = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const readoutRef = useRef<HTMLDivElement | null>(null);
  const coordsSpanRef = useRef<HTMLSpanElement | null>(null);
  const probSpanRef = useRef<HTMLSpanElement | null>(null);

  const [showTrain, setShowTrain] = useState(true);
  const [showVal, setShowVal] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // 1. Reuse offscreen canvas for heatmap interpolation (no DOM allocation per frame)
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const offscreen = offscreenRef.current;
    if (offscreen.width !== gridResolution || offscreen.height !== gridResolution) {
      offscreen.width = gridResolution;
      offscreen.height = gridResolution;
    }
    const offCtx = offscreen.getContext('2d');

    if (offCtx && gridPredictions.length === gridResolution * gridResolution) {
      const imgData = offCtx.createImageData(gridResolution, gridResolution);
      const data = imgData.data;

      for (let i = 0; i < gridPredictions.length; i++) {
        const p = gridPredictions[i];
        const distFromMid = Math.abs(p - 0.5) * 2;

        let r: number, g: number, b: number;
        if (p < 0.5) {
          const t = (0.5 - p) * 2;
          r = (16 * (1 - t) + 35 * t) | 0;
          g = (28 * (1 - t) + 150 * t) | 0;
          b = (54 * (1 - t) + 235 * t) | 0;
        } else {
          const t = (p - 0.5) * 2;
          r = (16 * (1 - t) + 245 * t) | 0;
          g = (28 * (1 - t) + 125 * t) | 0;
          b = (54 * (1 - t) + 45 * t) | 0;
        }

        const idx = i * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = (150 + distFromMid * 75) | 0;
      }

      offCtx.putImageData(imgData, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'low';
      ctx.drawImage(offscreen, 0, 0, width, height);
      ctx.restore();
    }

    // 2. Subtle coordinate grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gridSteps = 6;
    for (let i = 1; i < gridSteps; i++) {
      const gx = (width / gridSteps) * i;
      const gy = (height / gridSteps) * i;
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
    }
    ctx.stroke();

    // 3. Decision Boundary contour at P(class 1) = 0.5 using Marching Squares
    if (gridPredictions.length === gridResolution * gridResolution) {
      const cellW = width / (gridResolution - 1);
      const cellH = height / (gridResolution - 1);

      ctx.save();
      ctx.strokeStyle = '#59D9C5';
      ctx.lineWidth = 2.6;
      ctx.beginPath();

      const iso = 0.5;
      const interp = (v1: number, v2: number) => {
        const diff = v2 - v1;
        if (diff > -1e-6 && diff < 1e-6) return 0.5;
        return (iso - v1) / diff;
      };

      for (let r = 0; r < gridResolution - 1; r++) {
        const rowOffset = r * gridResolution;
        const nextRowOffset = (r + 1) * gridResolution;
        const y = r * cellH;

        for (let c = 0; c < gridResolution - 1; c++) {
          const tl = gridPredictions[rowOffset + c];
          const tr = gridPredictions[rowOffset + c + 1];
          const br = gridPredictions[nextRowOffset + c + 1];
          const bl = gridPredictions[nextRowOffset + c];

          let mask = 0;
          if (tl >= iso) mask |= 8;
          if (tr >= iso) mask |= 4;
          if (br >= iso) mask |= 2;
          if (bl >= iso) mask |= 1;

          if (mask === 0 || mask === 15) continue;

          const x = c * cellW;
          const topX = x + interp(tl, tr) * cellW;
          const topY = y;
          const rightX = x + cellW;
          const rightY = y + interp(tr, br) * cellH;
          const bottomX = x + interp(bl, br) * cellW;
          const bottomY = y + cellH;
          const leftX = x;
          const leftY = y + interp(tl, bl) * cellH;

          switch (mask) {
            case 1:
            case 14:
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(bottomX, bottomY);
              break;
            case 2:
            case 13:
              ctx.moveTo(bottomX, bottomY);
              ctx.lineTo(rightX, rightY);
              break;
            case 3:
            case 12:
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(rightX, rightY);
              break;
            case 4:
            case 11:
              ctx.moveTo(topX, topY);
              ctx.lineTo(rightX, rightY);
              break;
            case 5:
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(topX, topY);
              ctx.moveTo(bottomX, bottomY);
              ctx.lineTo(rightX, rightY);
              break;
            case 6:
            case 9:
              ctx.moveTo(topX, topY);
              ctx.lineTo(bottomX, bottomY);
              break;
            case 7:
            case 8:
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(topX, topY);
              break;
            case 10:
              ctx.moveTo(topX, topY);
              ctx.lineTo(rightX, rightY);
              ctx.moveTo(leftX, leftY);
              ctx.lineTo(bottomX, bottomY);
              break;
          }
        }
      }
      ctx.stroke();
      ctx.restore();
    }

    // 4. Render scatterplot points batched by style
    const invSpanX1 = width / (bounds.maxX1 - bounds.minX1);
    const invSpanX2 = height / (bounds.maxX2 - bounds.minX2);
    const radius = presentationMode ? 5.2 : 4.2;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      if (pt.split === 'train' && !showTrain) continue;
      if (pt.split === 'val' && !showVal) continue;

      const cx = (pt.x1 - bounds.minX1) * invSpanX1;
      const cy = (bounds.maxX2 - pt.x2) * invSpanX2;

      ctx.beginPath();
      ctx.arc(cx, cy, pt.split === 'train' ? radius : radius + 1, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#38BDF8' : '#FB923C';
      ctx.fill();
      ctx.lineWidth = pt.split === 'train' ? 1.3 : 1.8;
      ctx.strokeStyle = pt.split === 'train' ? '#070E1D' : '#FFFFFF';
      ctx.stroke();
    }
  }, [points, bounds, gridPredictions, gridResolution, showTrain, showVal, presentationMode]);

  // Direct DOM update on mouseMove avoids triggering a React re-render on every pixel move!
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const readout = readoutRef.current;
    if (!canvas || !readout || !coordsSpanRef.current || !probSpanRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const x1 = bounds.minX1 + relX * (bounds.maxX1 - bounds.minX1);
    const x2 = bounds.maxX2 - relY * (bounds.maxX2 - bounds.minX2);

    const col = Math.min(
      gridResolution - 1,
      Math.max(0, Math.round(relX * (gridResolution - 1)))
    );
    const row = Math.min(
      gridResolution - 1,
      Math.max(0, Math.round(relY * (gridResolution - 1)))
    );
    const prob =
      gridPredictions[row * gridResolution + col] !== undefined
        ? gridPredictions[row * gridResolution + col]
        : 0.5;

    readout.style.display = 'flex';
    coordsSpanRef.current.textContent = `x₁=${x1.toFixed(2)}, x₂=${x2.toFixed(2)}`;
    probSpanRef.current.textContent = `P(y=1) = ${(prob * 100).toFixed(1)}%`;
    probSpanRef.current.className =
      prob >= 0.5
        ? 'font-bold px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-300'
        : 'font-bold px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300';
  };

  const handleMouseLeave = () => {
    if (readoutRef.current) {
      readoutRef.current.style.display = 'none';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center gap-3 text-xs">
          <span className="inline-flex items-center gap-1.5 font-medium text-sky-400">
            <span className="w-3 h-3 rounded-full bg-[#38BDF8] border border-navy-950 inline-block" />
            Classe 0 (y=0)
          </span>
          <span className="inline-flex items-center gap-1.5 font-medium text-orange-400">
            <span className="w-3 h-3 rounded-full bg-[#FB923C] border border-navy-950 inline-block" />
            Classe 1 (y=1)
          </span>
          <span className="inline-flex items-center gap-1.5 font-medium text-turquoise">
            <span className="w-4 h-0.5 bg-turquoise inline-block" />
            Contour P(y=1) = 0.50
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowTrain((v) => !v)}
            className={`px-2 py-1 rounded-lg text-xs font-medium flex items-center gap-1 transition-all border ${
              showTrain
                ? 'bg-navy-800 text-slate-100 border-slate-700'
                : 'bg-navy-950/50 text-slate-500 border-slate-800'
            }`}
          >
            {showTrain ? <Eye className="w-3.5 h-3.5 text-turquoise" /> : <EyeOff className="w-3.5 h-3.5" />}
            Train
          </button>
          <button
            type="button"
            onClick={() => setShowVal((v) => !v)}
            className={`px-2 py-1 rounded-lg text-xs font-medium flex items-center gap-1 transition-all border ${
              showVal
                ? 'bg-navy-800 text-slate-100 border-white/40'
                : 'bg-navy-950/50 text-slate-500 border-slate-800'
            }`}
            title="Les points de validation ont un contour blanc"
          >
            {showVal ? <Eye className="w-3.5 h-3.5 text-violet-400" /> : <EyeOff className="w-3.5 h-3.5" />}
            Validation (cerclés blanc)
          </button>
        </div>
      </div>

      <div className="relative flex-1 min-h-[280px] rounded-xl overflow-hidden border border-slate-800 bg-navy-950 shadow-inner flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={480}
          height={380}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="w-full h-full object-contain cursor-crosshair"
        />

        {/* Probability colorbar legend */}
        <div className="absolute bottom-2.5 left-3 right-3 flex items-center justify-between px-3 py-1.5 rounded-lg bg-navy-950/85 border border-slate-800/80 text-[11px] pointer-events-none">
          <span className="text-sky-400 font-mono font-semibold">P(y=1) = 0.0</span>
          <div className="flex-1 mx-3 h-2 rounded-full bg-gradient-to-r from-[#38BDF8] via-[#59D9C5] to-[#FB923C]" />
          <span className="text-orange-400 font-mono font-semibold">P(y=1) = 1.0</span>
        </div>

        {/* Zero-lag cursor readout updated via ref */}
        <div
          ref={readoutRef}
          style={{ display: 'none' }}
          className="absolute top-2.5 right-2.5 px-2.5 py-1.5 rounded-lg bg-navy-950/95 border border-turquoise/40 text-xs font-mono items-center gap-2 pointer-events-none shadow-lg"
        >
          <Crosshair className="w-3.5 h-3.5 text-turquoise" />
          <span ref={coordsSpanRef} className="text-slate-300" />
          <span ref={probSpanRef} />
        </div>
      </div>
    </div>
  );
};

export const DecisionBoundaryCanvas = React.memo(DecisionBoundaryCanvasInner);
