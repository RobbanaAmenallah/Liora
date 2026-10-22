import React, { useState } from 'react';
import { LayerWeights, ModelConfig } from '@/types/ml';
import { Info } from 'lucide-react';

interface NetworkArchitectureSvgProps {
  config: ModelConfig;
  layerWeights: LayerWeights[];
  presentationMode?: boolean;
}

interface HoveredSynapse {
  layerIdx: number;
  fromIdx: number;
  toIdx: number;
  weight: number;
  bias: number;
}

const NetworkArchitectureSvgInner: React.FC<NetworkArchitectureSvgProps> = ({
  config,
  layerWeights,
  presentationMode = false,
}) => {
  const [hovered, setHovered] = useState<HoveredSynapse | null>(null);

  const layerSizes: number[] = [2];
  for (let i = 0; i < config.hiddenLayers; i++) {
    layerSizes.push(config.neuronsPerLayer);
  }
  layerSizes.push(1);

  const width = 620;
  const height = presentationMode ? 360 : 310;
  const padX = 64;
  const padY = 34;

  const numLayers = layerSizes.length;
  const layerX = layerSizes.map((_, i) =>
    numLayers === 1
      ? width / 2
      : padX + (i * (width - 2 * padX)) / (numLayers - 1)
  );

  const getNodeY = (layerIdx: number, nodeIdx: number) => {
    const count = layerSizes[layerIdx];
    if (count === 1) return height / 2;
    const usableH = height - 2 * padY;
    return padY + (nodeIdx * usableH) / (count - 1);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2 text-xs">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 text-turquoise font-medium">
            <span className="w-4 h-0.5 bg-turquoise inline-block" />
            Poids positif (w &gt; 0)
          </span>
          <span className="inline-flex items-center gap-1.5 text-orange-400 font-medium">
            <span className="w-4 h-0.5 bg-orange-400 inline-block" />
            Poids négatif (w &lt; 0)
          </span>
          <span className="text-slate-400">
            Épaisseur = |w| (magnitude)
          </span>
        </div>

        <span className="text-[11px] text-slate-400 flex items-center gap-1">
          <Info className="w-3.5 h-3.5 text-violet-400" />
          Survolez une connexion pour lire son poids exact
        </span>
      </div>

      <div className="relative flex-1 min-h-[250px] rounded-xl bg-navy-950 border border-slate-800 overflow-hidden flex items-center justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full select-none"
        >
          {/* Connections (Synapses) */}
          {layerSizes.slice(0, -1).map((fromCount, lIdx) => {
            const toCount = layerSizes[lIdx + 1];
            const lw = layerWeights[lIdx];

            const lines: React.ReactNode[] = [];
            for (let i = 0; i < fromCount; i++) {
              for (let j = 0; j < toCount; j++) {
                const x1 = layerX[lIdx];
                const y1 = getNodeY(lIdx, i);
                const x2 = layerX[lIdx + 1];
                const y2 = getNodeY(lIdx + 1, j);

                const w =
                  lw && lw.weights[i] && lw.weights[i][j] !== undefined
                    ? lw.weights[i][j]
                    : 0;
                const b =
                  lw && lw.biases && lw.biases[j] !== undefined
                    ? lw.biases[j]
                    : 0;

                const absW = Math.abs(w);
                const strokeW = Math.min(5.5, Math.max(0.8, absW * 2.1));
                const opacity = Math.min(0.95, Math.max(0.24, absW * 0.55));
                const color = w >= 0 ? '#59D9C5' : '#FB923C';

                const isHovered =
                  hovered &&
                  hovered.layerIdx === lIdx &&
                  hovered.fromIdx === i &&
                  hovered.toIdx === j;

                lines.push(
                  <line
                    key={`syn-${lIdx}-${i}-${j}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={isHovered ? '#FFFFFF' : color}
                    strokeWidth={isHovered ? strokeW + 2 : strokeW}
                    strokeOpacity={isHovered ? 1 : opacity}
                    strokeLinecap="round"
                    className="cursor-pointer"
                    onMouseEnter={() =>
                      setHovered({
                        layerIdx: lIdx,
                        fromIdx: i,
                        toIdx: j,
                        weight: w,
                        bias: b,
                      })
                    }
                    onMouseLeave={() => setHovered(null)}
                  />
                );
              }
            }
            return <g key={`layer-lines-${lIdx}`}>{lines}</g>;
          })}

          {/* Neurons (Nodes) */}
          {layerSizes.map((count, lIdx) => {
            const isInput = lIdx === 0;
            const isOutput = lIdx === layerSizes.length - 1;
            const radius = count > 12 ? 9 : count > 8 ? 11 : 14;

            return (
              <g key={`layer-nodes-${lIdx}`}>
                <text
                  x={layerX[lIdx]}
                  y={16}
                  textAnchor="middle"
                  className="fill-slate-400 text-[10px] font-semibold uppercase tracking-wider"
                >
                  {isInput
                    ? 'Entrées (2)'
                    : isOutput
                    ? 'Sortie σ (1)'
                    : `Cachée ${lIdx} (${count})`}
                </text>

                {Array.from({ length: count }).map((_, nIdx) => {
                  const cx = layerX[lIdx];
                  const cy = getNodeY(lIdx, nIdx);
                  const lwPrev = lIdx > 0 ? layerWeights[lIdx - 1] : null;
                  const bias =
                    lwPrev && lwPrev.biases[nIdx] !== undefined
                      ? lwPrev.biases[nIdx]
                      : 0;

                  const strokeColor = isInput
                    ? '#38BDF8'
                    : isOutput
                    ? '#FB923C'
                    : '#986DC7';

                  return (
                    <g key={`node-${lIdx}-${nIdx}`}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={radius}
                        fill="#0D1830"
                        stroke={strokeColor}
                        strokeWidth={2.4}
                      />
                      {count <= 10 && (
                        <text
                          x={cx}
                          y={cy + 3.5}
                          textAnchor="middle"
                          className="fill-slate-200 text-[9px] font-mono font-semibold pointer-events-none"
                        >
                          {isInput
                            ? `x${nIdx + 1}`
                            : isOutput
                            ? 'ŷ'
                            : bias >= 0
                            ? `+${bias.toFixed(1)}`
                            : bias.toFixed(1)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>

        {hovered && (
          <div className="absolute top-2.5 left-2.5 px-3 py-2 rounded-xl bg-navy-900/95 border border-turquoise/50 shadow-xl text-xs font-mono pointer-events-none">
            <div className="text-slate-300 font-sans font-semibold mb-0.5">
              Connexion : Couche {hovered.layerIdx} (neurone {hovered.fromIdx + 1}) → Couche{' '}
              {hovered.layerIdx + 1} (neurone {hovered.toIdx + 1})
            </div>
            <div className="flex items-center gap-3">
              <span
                className={
                  hovered.weight >= 0 ? 'text-turquoise font-bold' : 'text-orange-400 font-bold'
                }
              >
                w = {hovered.weight >= 0 ? '+' : ''}
                {hovered.weight.toFixed(4)}
              </span>
              <span className="text-violet-300">
                b_dest = {hovered.bias >= 0 ? '+' : ''}
                {hovered.bias.toFixed(4)}
              </span>
            </div>
          </div>
        )}

        <div className="absolute bottom-2 right-3 text-[10px] text-slate-400 italic">
          Visualisation statique des paramètres appris (poids w et biais b) du modèle TensorFlow.js
        </div>
      </div>
    </div>
  );
};

export const NetworkArchitectureSvg = React.memo(NetworkArchitectureSvgInner);
