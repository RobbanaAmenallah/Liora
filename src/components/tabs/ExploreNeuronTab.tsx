import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { NeuronActivationType } from '@/types/ml';
import { FrenchTooltip } from '@/components/ui/FrenchTooltip';
import {
  Sliders,
  Zap,
  Sparkles,
  RotateCcw,
  CheckCircle2,
  BookOpen,
  ArrowRight,
} from 'lucide-react';

interface ExploreNeuronTabProps {
  presentationMode?: boolean;
}

function computeActivation(z: number, type: NeuronActivationType): number {
  switch (type) {
    case 'linear':
      return z;
    case 'relu':
      return Math.max(0, z);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-z));
    case 'tanh':
      return Math.tanh(z);
  }
}

export const ExploreNeuronTab: React.FC<ExploreNeuronTabProps> = ({
  presentationMode = false,
}) => {
  const [x1, setX1] = useState<number>(1.2);
  const [x2, setX2] = useState<number>(-0.8);
  const [w1, setW1] = useState<number>(1.5);
  const [w2, setW2] = useState<number>(-1.0);
  const [b, setB] = useState<number>(-0.3);
  const [activation, setActivation] = useState<NeuronActivationType>('relu');
  const [lastChanged, setLastChanged] = useState<string>(
    'Déplacez un curseur (entrée, poids, biais ou fonction d’activation) pour observer son effet immédiat.'
  );
  const [predictAnswerShown, setPredictAnswerShown] = useState(false);

  const prod1 = w1 * x1;
  const prod2 = w2 * x2;
  const z = prod1 + prod2 + b;
  const a = computeActivation(z, activation);

  const handleReset = () => {
    setX1(1.2);
    setX2(-0.8);
    setW1(1.5);
    setW2(-1.0);
    setB(-0.3);
    setActivation('relu');
    setLastChanged('Valeurs réinitialisées aux paramètres pédagogiques par défaut.');
  };

  // Generate mini SVG curve for the selected activation function over z in [-4, 4]
  const curvePoints: string[] = [];
  const curveW = 220;
  const curveH = 110;
  for (let i = 0; i <= 60; i++) {
    const zv = -4 + (i / 60) * 8;
    const av = computeActivation(zv, activation);
    const px = (i / 60) * curveW;
    // Map y range [-2, 2] into [curveH - 12, 12]
    const clampedY = Math.max(-2, Math.min(2, av));
    const py = curveH / 2 - (clampedY / 2) * (curveH / 2 - 12);
    curvePoints.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }

  const currentClampedZ = Math.max(-4, Math.min(4, z));
  const currentPx = ((currentClampedZ + 4) / 8) * curveW;
  const currentClampedA = Math.max(-2, Math.min(2, a));
  const currentPy = curveH / 2 - (currentClampedA / 2) * (curveH / 2 - 12);

  return (
    <div className="space-y-6">
      {/* Top Pedagogical Banner */}
      <div className="p-4 rounded-2xl bg-gradient-to-r from-navy-800 via-navy-800/90 to-violet-950/50 border border-turquoise/30 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-turquoise">
            <Sparkles className="w-4 h-4" />
            Étape 1 — Anatomie mathématique d’un neurone formel
          </div>
          <h2 className="text-lg md:text-xl font-bold text-white">
            Du modèle linéaire de scikit-learn à la brique élémentaire du Deep Learning
          </h2>
          <p className="text-xs md:text-sm text-slate-300 max-w-4xl">
            Un neurone artificiel réalise deux opérations simples : une{' '}
            <strong className="text-turquoise">combinaison linéaire pondérée</strong>{' '}
            (<code className="text-slate-100">z = w₁x₁ + w₂x₂ + b</code>), suivie d’une{' '}
            <strong className="text-violet-400">fonction d’activation non-linéaire</strong>{' '}
            (<code className="text-slate-100">a = φ(z)</code>).
          </p>
        </div>

        <button
          type="button"
          onClick={handleReset}
          className="px-3.5 py-2 rounded-xl bg-navy-900 hover:bg-navy-700 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-2 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5 text-turquoise" />
          Réinitialiser le neurone
        </button>
      </div>

      {/* Main Grid: Controls + Interactive Animated Flow Diagram */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Left column: Sliders & Activation Selector */}
        {!presentationMode && (
          <div className="xl:col-span-4 space-y-4">
            {/* Inputs Card */}
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-sky-400 flex items-center gap-1.5">
                  <Sliders className="w-4 h-4" />
                  1. Variables d’entrée (Features x₁, x₂)
                </h3>
                <FrenchTooltip
                  term="Inputs (x₁, x₂)"
                  explanation="Les caractéristiques numériques d’une observation (équivalent d’une ligne de votre matrice X dans scikit-learn)."
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Entrée x₁</span>
                  <span className="font-mono font-bold text-sky-400">{x1.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.1}
                  value={x1}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setX1(val);
                    setLastChanged(
                      `Vous avez modifié l’entrée x₁ à ${val.toFixed(2)}. La contribution w₁·x₁ vaut maintenant ${(w1 * val).toFixed(2)}.`
                    );
                  }}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Entrée x₂</span>
                  <span className="font-mono font-bold text-sky-400">{x2.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.1}
                  value={x2}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setX2(val);
                    setLastChanged(
                      `Vous avez modifié l’entrée x₂ à ${val.toFixed(2)}. La contribution w₂·x₂ vaut maintenant ${(w2 * val).toFixed(2)}.`
                    );
                  }}
                  className="w-full"
                />
              </div>
            </div>

            {/* Weights & Bias Card */}
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-turquoise flex items-center gap-1.5">
                  <Zap className="w-4 h-4" />
                  2. Paramètres appris (Poids w₁, w₂ et Biais b)
                </h3>
                <FrenchTooltip
                  term="Weights (w₁, w₂) & Bias (b)"
                  explanation="Les paramètres ajustés pendant l’entraînement par descente de gradient. Un poids positif amplifie l’entrée, un poids négatif l’inhibe."
                  sklearnNote="Équivalent de clf.coef_ et clf.intercept_."
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Poids synaptique w₁</span>
                  <span
                    className={`font-mono font-bold ${
                      w1 >= 0 ? 'text-turquoise' : 'text-orange-400'
                    }`}
                  >
                    {w1 >= 0 ? `+${w1.toFixed(2)}` : w1.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.1}
                  value={w1}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setW1(val);
                    setLastChanged(
                      `Poids w₁ réglé à ${val.toFixed(2)} (${
                        val >= 0 ? 'excitateur' : 'inhibiteur'
                      }). L’épaisseur et la couleur de la connexion x₁ → Σ s’adaptent en direct.`
                    );
                  }}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Poids synaptique w₂</span>
                  <span
                    className={`font-mono font-bold ${
                      w2 >= 0 ? 'text-turquoise' : 'text-orange-400'
                    }`}
                  >
                    {w2 >= 0 ? `+${w2.toFixed(2)}` : w2.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.1}
                  value={w2}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setW2(val);
                    setLastChanged(
                      `Poids w₂ réglé à ${val.toFixed(2)} (${
                        val >= 0 ? 'excitateur' : 'inhibiteur'
                      }). Le produit w₂·x₂ contribue pour ${(val * x2).toFixed(2)} à la somme z.`
                    );
                  }}
                  className="w-full"
                />
              </div>

              <div className="space-y-2 pt-1 border-t border-slate-800/80">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Biais (Ordonnée à l’origine) b</span>
                  <span className="font-mono font-bold text-violet-400">
                    {b >= 0 ? `+${b.toFixed(2)}` : b.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.1}
                  value={b}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setB(val);
                    setLastChanged(
                      `Biais b décalé à ${val.toFixed(2)}. Il translate le seuil d’activation indépendamment des entrées x₁ et x₂.`
                    );
                  }}
                  className="w-full"
                />
              </div>
            </div>

            {/* Activation Function Selector */}
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-violet-400">
                  3. Fonction d’activation φ(z)
                </h3>
                <FrenchTooltip
                  term="Activation Function"
                  explanation="Transforme la somme pondérée z en sortie finale a. Sans non-linéarité (comme ReLU, Sigmoid ou Tanh), un réseau profond se réduit à une simple régression linéaire !"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                {(
                  [
                    { id: 'linear', label: 'Linear', desc: 'φ(z) = z' },
                    { id: 'relu', label: 'ReLU', desc: 'max(0, z)' },
                    { id: 'sigmoid', label: 'Sigmoid', desc: '1 / (1 + e⁻ᶻ)' },
                    { id: 'tanh', label: 'Tanh', desc: 'tanh(z) ∈ [-1,1]' },
                  ] as const
                ).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setActivation(item.id);
                      setLastChanged(
                        `Activation changée vers ${item.label} (${item.desc}). Pour z = ${z.toFixed(2)}, la sortie a vaut maintenant ${computeActivation(z, item.id).toFixed(3)}.`
                      );
                    }}
                    className={`p-2.5 rounded-xl border text-left transition-all ${
                      activation === item.id
                        ? 'bg-violet-500/20 border-violet-400 text-white shadow-md'
                        : 'bg-navy-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="text-xs font-bold">{item.label}</div>
                    <div className="text-[11px] font-mono opacity-80">{item.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Right column: Animated Interactive SVG Neuron + Substituted Equations */}
        <div className={presentationMode ? 'xl:col-span-12 space-y-4' : 'xl:col-span-8 space-y-4'}>
          {presentationMode && (
            <div className="p-3 rounded-xl bg-navy-900 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-4 text-xs">
                <label className="flex items-center gap-2">
                  <span className="text-sky-400 font-bold">x₁={x1.toFixed(1)}</span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.1}
                    value={x1}
                    onChange={(e) => setX1(parseFloat(e.target.value))}
                    className="w-24"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-sky-400 font-bold">x₂={x2.toFixed(1)}</span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.1}
                    value={x2}
                    onChange={(e) => setX2(parseFloat(e.target.value))}
                    className="w-24"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-turquoise font-bold">w₁={w1.toFixed(1)}</span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.1}
                    value={w1}
                    onChange={(e) => setW1(parseFloat(e.target.value))}
                    className="w-24"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-turquoise font-bold">w₂={w2.toFixed(1)}</span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.1}
                    value={w2}
                    onChange={(e) => setW2(parseFloat(e.target.value))}
                    className="w-24"
                  />
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-violet-400 font-bold">b={b.toFixed(1)}</span>
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.1}
                    value={b}
                    onChange={(e) => setB(parseFloat(e.target.value))}
                    className="w-24"
                  />
                </label>
              </div>
              <div className="flex gap-1.5">
                {(['linear', 'relu', 'sigmoid', 'tanh'] as const).map((act) => (
                  <button
                    key={act}
                    type="button"
                    onClick={() => setActivation(act)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-bold uppercase ${
                      activation === act
                        ? 'bg-turquoise text-navy-950'
                        : 'bg-navy-800 text-slate-300'
                    }`}
                  >
                    {act}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* SVG Flow Diagram */}
          <div className="p-5 rounded-2xl bg-navy-900/95 border border-slate-800 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider">
                Flux de Propagation Avant (Entrées → Somme Pondérée z → Activation a)
              </h3>
              <span className="text-xs px-2.5 py-1 rounded-full bg-turquoise/10 text-turquoise font-mono">
                Sortie finale a = {a.toFixed(4)}
              </span>
            </div>

            <div className="w-full overflow-x-auto">
              <svg viewBox="0 0 820 310" className="w-full min-w-[660px] h-auto select-none">
                <defs>
                  <linearGradient id="flowGradPos" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#38BDF8" />
                    <stop offset="100%" stopColor="#59D9C5" />
                  </linearGradient>
                  <linearGradient id="flowGradNeg" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#38BDF8" />
                    <stop offset="100%" stopColor="#FB923C" />
                  </linearGradient>
                </defs>

                {/* Connection x1 -> Sum */}
                <line
                  x1={115}
                  y1={85}
                  x2={340}
                  y2={155}
                  stroke={w1 >= 0 ? '#59D9C5' : '#FB923C'}
                  strokeWidth={Math.max(1.5, Math.min(8, Math.abs(w1) * 2.5))}
                  strokeOpacity={0.8}
                />
                <motion.circle
                  r={5}
                  fill="#FFFFFF"
                  animate={{
                    cx: [115, 340],
                    cy: [85, 155],
                  }}
                  transition={{
                    duration: 1.8,
                    repeat: Infinity,
                    ease: 'linear',
                  }}
                />

                {/* Weight 1 badge */}
                <rect
                  x={175}
                  y={92}
                  width={110}
                  height={38}
                  rx={8}
                  fill="#070E1D"
                  stroke={w1 >= 0 ? '#59D9C5' : '#FB923C'}
                  strokeWidth={1.5}
                />
                <text x={230} y={108} textAnchor="middle" className="fill-slate-300 text-[10px] font-mono">
                  w₁ = {w1 >= 0 ? `+${w1.toFixed(2)}` : w1.toFixed(2)}
                </text>
                <text x={230} y={123} textAnchor="middle" className="fill-turquoise text-[11px] font-mono font-bold">
                  w₁·x₁ = {prod1 >= 0 ? `+${prod1.toFixed(2)}` : prod1.toFixed(2)}
                </text>

                {/* Connection x2 -> Sum */}
                <line
                  x1={115}
                  y1={225}
                  x2={340}
                  y2={155}
                  stroke={w2 >= 0 ? '#59D9C5' : '#FB923C'}
                  strokeWidth={Math.max(1.5, Math.min(8, Math.abs(w2) * 2.5))}
                  strokeOpacity={0.8}
                />
                <motion.circle
                  r={5}
                  fill="#FFFFFF"
                  animate={{
                    cx: [115, 340],
                    cy: [225, 155],
                  }}
                  transition={{
                    duration: 1.8,
                    repeat: Infinity,
                    ease: 'linear',
                    delay: 0.4,
                  }}
                />

                {/* Weight 2 badge */}
                <rect
                  x={175}
                  y={180}
                  width={110}
                  height={38}
                  rx={8}
                  fill="#070E1D"
                  stroke={w2 >= 0 ? '#59D9C5' : '#FB923C'}
                  strokeWidth={1.5}
                />
                <text x={230} y={196} textAnchor="middle" className="fill-slate-300 text-[10px] font-mono">
                  w₂ = {w2 >= 0 ? `+${w2.toFixed(2)}` : w2.toFixed(2)}
                </text>
                <text x={230} y={211} textAnchor="middle" className="fill-orange-400 text-[11px] font-mono font-bold">
                  w₂·x₂ = {prod2 >= 0 ? `+${prod2.toFixed(2)}` : prod2.toFixed(2)}
                </text>

                {/* Bias connection into Sum */}
                <line
                  x1={380}
                  y1={42}
                  x2={380}
                  y2={105}
                  stroke="#986DC7"
                  strokeWidth={2.5}
                  strokeDasharray="4 3"
                />
                <rect
                  x={325}
                  y={12}
                  width={110}
                  height={30}
                  rx={8}
                  fill="#070E1D"
                  stroke="#986DC7"
                  strokeWidth={1.5}
                />
                <text x={380} y={31} textAnchor="middle" className="fill-violet-300 text-xs font-mono font-bold">
                  Biais b = {b >= 0 ? `+${b.toFixed(2)}` : b.toFixed(2)}
                </text>

                {/* Input Node x1 */}
                <circle cx={75} cy={85} r={38} fill="#070E1D" stroke="#38BDF8" strokeWidth={3} />
                <text x={75} y={78} textAnchor="middle" className="fill-slate-400 text-xs font-bold">
                  Entrée x₁
                </text>
                <text x={75} y={98} textAnchor="middle" className="fill-sky-400 text-sm font-mono font-bold">
                  {x1.toFixed(2)}
                </text>

                {/* Input Node x2 */}
                <circle cx={75} cy={225} r={38} fill="#070E1D" stroke="#38BDF8" strokeWidth={3} />
                <text x={75} y={218} textAnchor="middle" className="fill-slate-400 text-xs font-bold">
                  Entrée x₂
                </text>
                <text x={75} y={238} textAnchor="middle" className="fill-sky-400 text-sm font-mono font-bold">
                  {x2.toFixed(2)}
                </text>

                {/* Weighted Sum Node (z) */}
                <circle cx={380} cy={155} r={48} fill="#0D1830" stroke="#59D9C5" strokeWidth={3.5} />
                <text x={380} y={138} textAnchor="middle" className="fill-slate-300 text-xs font-bold">
                  Somme Σ (z)
                </text>
                <text x={380} y={162} textAnchor="middle" className="fill-turquoise text-base font-mono font-extrabold">
                  {z >= 0 ? `+${z.toFixed(2)}` : z.toFixed(2)}
                </text>
                <text x={380} y={180} textAnchor="middle" className="fill-slate-400 text-[10px] font-mono">
                  w₁x₁ + w₂x₂ + b
                </text>

                {/* Connection Sum -> Activation */}
                <line
                  x1={428}
                  y1={155}
                  x2={515}
                  y2={155}
                  stroke="#59D9C5"
                  strokeWidth={3}
                />
                <motion.circle
                  r={5}
                  fill="#59D9C5"
                  animate={{
                    cx: [428, 515],
                    cy: [155, 155],
                  }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: 'linear',
                  }}
                />

                {/* Activation Box with live mini-plot */}
                <rect
                  x={515}
                  y={90}
                  width={145}
                  height={130}
                  rx={16}
                  fill="#070E1D"
                  stroke="#986DC7"
                  strokeWidth={2.5}
                />
                <text x={587} y={110} textAnchor="middle" className="fill-violet-300 text-xs font-bold uppercase">
                  Activation : {activation}
                </text>
                <svg x={527} y={118} width={120} height={90} viewBox={`0 0 ${curveW} ${curveH}`}>
                  <line x1={0} y1={curveH / 2} x2={curveW} y2={curveH / 2} stroke="#1B2F59" strokeWidth={1.5} />
                  <line x1={curveW / 2} y1={0} x2={curveW / 2} y2={curveH} stroke="#1B2F59" strokeWidth={1.5} />
                  <polyline
                    fill="none"
                    stroke="#986DC7"
                    strokeWidth={3.5}
                    points={curvePoints.join(' ')}
                  />
                  <circle
                    cx={currentPx}
                    cy={currentPy}
                    r={6.5}
                    fill="#59D9C5"
                    stroke="#070E1D"
                    strokeWidth={2}
                  />
                </svg>

                {/* Connection Activation -> Output a */}
                <line
                  x1={660}
                  y1={155}
                  x2={715}
                  y2={155}
                  stroke="#986DC7"
                  strokeWidth={3}
                />
                <motion.circle
                  r={5}
                  fill="#FB923C"
                  animate={{
                    cx: [660, 715],
                    cy: [155, 155],
                  }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: 'linear',
                  }}
                />

                {/* Final Output Node (a) */}
                <circle
                  cx={760}
                  cy={155}
                  r={44}
                  fill="#0D1830"
                  stroke="#FB923C"
                  strokeWidth={3.5}
                />
                <text x={760} y={140} textAnchor="middle" className="fill-slate-300 text-xs font-bold">
                  Sortie a
                </text>
                <text x={760} y={163} textAnchor="middle" className="fill-orange-400 text-base font-mono font-extrabold">
                  {a.toFixed(3)}
                </text>
                <text x={760} y={180} textAnchor="middle" className="fill-slate-400 text-[10px] font-mono">
                  φ(z)
                </text>
              </svg>
            </div>

            {/* Live Substituted Formula Box */}
            <div className="mt-4 p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-2 font-mono text-xs md:text-sm">
              <div className="flex flex-wrap items-center gap-2 text-slate-300">
                <span className="text-turquoise font-bold">1. Somme pondérée (Pré-activation) :</span>
                <span>z = w₁·x₁ + w₂·x₂ + b =</span>
                <span className="text-sky-300">
                  ({w1.toFixed(2)} × {x1.toFixed(2)})
                </span>
                <span>+</span>
                <span className="text-sky-300">
                  ({w2.toFixed(2)} × {x2.toFixed(2)})
                </span>
                <span>+</span>
                <span className="text-violet-300">({b.toFixed(2)})</span>
                <span>=</span>
                <span className="px-2 py-0.5 rounded bg-turquoise/15 text-turquoise font-bold">
                  {z.toFixed(3)}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-slate-300">
                <span className="text-violet-400 font-bold">2. Activation non-linéaire :</span>
                <span>
                  a = {activation.toUpperCase()}(z) = {activation.toUpperCase()}({z.toFixed(3)}) =
                </span>
                <span className="px-2 py-0.5 rounded bg-orange-500/20 text-orange-300 font-bold">
                  {a.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          {/* Dynamic Educational Explanation + Mini Predict Challenge */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-turquoise/30 space-y-2">
              <div className="flex items-center gap-2 text-xs font-bold text-turquoise uppercase tracking-wider">
                <BookOpen className="w-4 h-4" />
                Journal d’analyse en direct
              </div>
              <p className="text-xs md:text-sm text-slate-200 leading-relaxed">{lastChanged}</p>
              <div className="pt-2 border-t border-slate-800 text-xs text-slate-400">
                <strong className="text-slate-200">Interprétation géométrique :</strong> Dans le plan{' '}
                <code>(x₁, x₂)</code>, l’équation <code>w₁x₁ + w₂x₂ + b = 0</code> définit une droite.
                Le vecteur <code>(w₁, w₂)</code> est normal (perpendiculaire) à cette droite !
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-navy-900/90 border border-violet-500/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-violet-400 uppercase tracking-wider">
                  Défi « Prédisez avant de manipuler »
                </span>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300">
                  Intuition ReLU
                </span>
              </div>
              <p className="text-xs text-slate-200">
                Avec l’activation <strong>ReLU</strong> sélectionnée, que se passe-t-il sur la sortie{' '}
                <code>a</code> si vous diminuez le biais <code>b</code> jusqu’à ce que <code>z &lt; 0</code> ?
              </p>
              {!predictAnswerShown ? (
                <button
                  type="button"
                  onClick={() => setPredictAnswerShown(true)}
                  className="mt-1 px-3 py-1.5 rounded-lg bg-violet-600/30 hover:bg-violet-600/50 border border-violet-400/40 text-xs font-semibold text-violet-200 inline-flex items-center gap-1.5 transition-colors"
                >
                  Vérifier ma prédiction <ArrowRight className="w-3.5 h-3.5" />
                </button>
              ) : (
                <div className="p-2.5 rounded-xl bg-navy-950/90 border border-turquoise/40 text-xs text-slate-200 flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-turquoise shrink-0 mt-0.5" />
                  <div>
                    <strong>Exact !</strong> Dès que <code>z &lt; 0</code>, ReLU coupe le signal et renvoie{' '}
                    <code>a = 0</code> (neurone inactif). C’est ce mécanisme d’interrupteur qui permet à une
                    couche cachée d’activer différents neurones selon les régions de l’espace !
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
