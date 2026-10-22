import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  ActivationType,
  DatasetType,
  GeneratedDataset,
  ModelConfig,
} from '@/types/ml';
import { generateDataset } from '@/lib/datasets';
import {
  buildNormalizedGridTensor,
  countTrainableParameters,
  createNeuralNetwork,
  ensureTfBackend,
  evaluateSplit,
  pointsToTensors,
} from '@/lib/tfModel';
import {
  Trophy,
  Play,
  CheckCircle2,
  Award,
  Sparkles,
  RotateCcw,
  ArrowRight,
  Layers,
} from 'lucide-react';

interface SimpleMission {
  id: 1 | 2 | 3;
  title: string;
  badge: string;
  contextFr: string;
  datasetType: DatasetType;
  samples: number;
  noise: number;
  seed: number;
  defaultLayers: number;
  defaultNeurons: number;
  defaultActivation: ActivationType;
  targetValAcc: number;
  maxParams: number;
  questionFr: string;
  optionsFr: string[];
  correctIdx: number;
  explanationFr: string;
}

const MISSIONS: SimpleMission[] = [
  {
    id: 1,
    title: 'Mission 1 : Vaincre la non-linéarité (Two Moons)',
    badge: 'Frontière Courbe',
    contextFr:
      'Un modèle linéaire (0 couche cachée) trace une simple ligne droite et plafonne vers 86%. Choisissez une architecture simple capable de dépasser 92% de précision en validation.',
    datasetType: 'moons',
    samples: 300,
    noise: 0.14,
    seed: 101,
    defaultLayers: 0,
    defaultNeurons: 6,
    defaultActivation: 'relu',
    targetValAcc: 0.92,
    maxParams: 80,
    questionFr: 'Pourquoi faut-il au moins 1 couche cachée avec ReLU ou Tanh sur Two Moons ?',
    optionsFr: [
      'Pour plier l’espace et créer une frontière de décision non-linéaire',
      'Uniquement pour accélérer le processeur graphique',
      'Pour transformer les étiquettes en texte',
    ],
    correctIdx: 0,
    explanationFr:
      'Exact ! Sans couche cachée non-linéaire, le réseau reste une régression logistique (une droite). 1 couche cachée de 6 à 8 neurones suffit largement !',
  },
  {
    id: 2,
    title: 'Mission 2 : Entourer le cercle central (Circles)',
    badge: 'Topologie Fermée',
    contextFr:
      'Sur des cercles concentriques, une ligne droite coupe le disque en deux (~50% de réussite, comme un pile ou face). Construisez un réseau qui entoure le cercle central (≥ 92%).',
    datasetType: 'circles',
    samples: 320,
    noise: 0.1,
    seed: 202,
    defaultLayers: 0,
    defaultNeurons: 6,
    defaultActivation: 'relu',
    targetValAcc: 0.92,
    maxParams: 90,
    questionFr:
      'Combien de neurones cachés (droites) faut-il au minimum en 2D pour fermer une zone autour d’un cercle ?',
    optionsFr: [
      '1 seul neurone suffit pour faire un cercle',
      'Au moins 3 ou 4 neurones (qui combinent leurs demi-plans pour fermer un polygone autour du centre)',
      'Il faut obligatoirement 100 couches cachées',
    ],
    correctIdx: 1,
    explanationFr:
      'Exactement ! En 2D, 3 droites forment un triangle fermé. Dès 4 à 8 neurones dans 1 couche cachée, la frontière encercle proprement la classe centrale.',
  },
  {
    id: 3,
    title: 'Mission 3 : Éviter le Sur-apprentissage (Overfitting)',
    badge: 'Sobriété & Bruit',
    contextFr:
      'Les données sont très bruitées ! Le réseau actuel est beaucoup trop gros (4 couches de 16 neurones = 881 paramètres) et mémorise le bruit. Simplifiez-le (≤ 65 paramètres) tout en gardant ≥ 84% en validation.',
    datasetType: 'moons',
    samples: 200,
    noise: 0.33,
    seed: 303,
    defaultLayers: 4,
    defaultNeurons: 16,
    defaultActivation: 'relu',
    targetValAcc: 0.84,
    maxParams: 65,
    questionFr:
      'Que se passe-t-il quand un réseau possède beaucoup trop de paramètres sur un petit jeu de données bruité ?',
    optionsFr: [
      'Il fait de l’Overfitting : il mémorise le bruit d’entraînement et généralise mal en validation',
      'Il devient automatiquement un modèle linéaire',
      'La fonction de coût devient négative',
    ],
    correctIdx: 0,
    explanationFr:
      'Parfait ! Réduire le réseau à 1 seule couche de 6 neurones (~25 paramètres) empêche de mémoriser les anomalies et stabilise la Validation Accuracy.',
  },
];

interface MissionResult {
  points: number;
  passed: boolean;
  trainAcc: number;
  valAcc: number;
  params: number;
  feedback: string;
}

export const ChallengeTab: React.FC = () => {
  const [missionIdx, setMissionIdx] = useState<number>(0);
  const mission = MISSIONS[missionIdx];

  // Minimal, clean state for the current mission
  const [hiddenLayers, setHiddenLayers] = useState<number>(mission.defaultLayers);
  const [neurons, setNeurons] = useState<number>(mission.defaultNeurons);
  const [activation, setActivation] = useState<ActivationType>(mission.defaultActivation);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [trainAcc, setTrainAcc] = useState<number | null>(null);
  const [valAcc, setValAcc] = useState<number | null>(null);
  const [paramsCount, setParamsCount] = useState<number>(0);

  const [results, setResults] = useState<Record<number, MissionResult>>({});
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const datasetRef = useRef<GeneratedDataset>(
    generateDataset(mission.datasetType, mission.samples, mission.noise, 0.8, mission.seed)
  );

  // Compute parameter count preview dynamically without heavy rendering
  useEffect(() => {
    if (hiddenLayers === 0) {
      setParamsCount(3); // 2 weights + 1 bias
    } else {
      // First layer: 2 * N + N
      let total = 2 * neurons + neurons;
      // Subsequent hidden layers: N * N + N
      for (let i = 1; i < hiddenLayers; i++) {
        total += neurons * neurons + neurons;
      }
      // Output layer: N * 1 + 1
      total += neurons + 1;
      setParamsCount(total);
    }
  }, [hiddenLayers, neurons]);

  // Draw clean compact mini-preview of dataset & boundary
  const drawMiniCanvas = (ds: GeneratedDataset, gridPreds?: Float32Array) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const res = 28;
    if (gridPreds && gridPreds.length === res * res) {
      const off = document.createElement('canvas');
      off.width = res;
      off.height = res;
      const offCtx = off.getContext('2d');
      if (offCtx) {
        const img = offCtx.createImageData(res, res);
        for (let i = 0; i < gridPreds.length; i++) {
          const p = gridPreds[i];
          const idx = i * 4;
          if (p < 0.5) {
            img.data[idx] = 35;
            img.data[idx + 1] = 150;
            img.data[idx + 2] = 235;
            img.data[idx + 3] = 95;
          } else {
            img.data[idx] = 245;
            img.data[idx + 1] = 125;
            img.data[idx + 2] = 45;
            img.data[idx + 3] = 95;
          }
        }
        offCtx.putImageData(img, 0, 0);
        ctx.drawImage(off, 0, 0, w, h);
      }
    }

    const { bounds, points } = ds;
    for (const pt of points) {
      const cx = ((pt.x1 - bounds.minX1) / (bounds.maxX1 - bounds.minX1)) * w;
      const cy = ((bounds.maxX2 - pt.x2) / (bounds.maxX2 - bounds.minX2)) * h;
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? '#38BDF8' : '#FB923C';
      ctx.fill();
    }
  };

  // Reset mission state when switching mission tab
  useEffect(() => {
    const m = MISSIONS[missionIdx];
    const ds = generateDataset(m.datasetType, m.samples, m.noise, 0.8, m.seed);
    datasetRef.current = ds;
    setHiddenLayers(m.defaultLayers);
    setNeurons(m.defaultNeurons);
    setActivation(m.defaultActivation);
    setSelectedAnswer(null);
    setTrainAcc(null);
    setValAcc(null);
    setProgress(0);
    setTimeout(() => drawMiniCanvas(ds), 20);
  }, [missionIdx]);

  // One-click Test & Validate Mission
  const handleRunAndEvaluate = async () => {
    setIsRunning(true);
    setProgress(10);

    await ensureTfBackend();
    const ds = datasetRef.current;

    const cfg: ModelConfig = {
      hiddenLayers,
      neuronsPerLayer: neurons,
      activation,
      learningRate: 0.04,
      epochs: 45,
      batchSize: 32,
    };

    const model = createNeuralNetwork(cfg, mission.seed);
    const actualParams = countTrainableParameters(model);
    setParamsCount(actualParams);

    const tr = pointsToTensors(ds.trainPoints, ds.stats);
    const va = pointsToTensors(ds.valPoints, ds.stats);
    const gridTensor = buildNormalizedGridTensor(ds.bounds, ds.stats, 28);

    try {
      // Train in 3 fast chunks of 15 epochs (total 45 epochs -> takes ~0.3 seconds)
      for (let step = 1; step <= 3; step++) {
        await model.fit(tr.xs, tr.ys, {
          epochs: 15,
          batchSize: 32,
          shuffle: true,
          verbose: 0,
        });
        setProgress(step * 33);
        await new Promise<void>((r) => requestAnimationFrame(() => r()));
      }

      const trEval = evaluateSplit(model, tr.xs, tr.ys);
      const vaEval = evaluateSplit(model, va.xs, va.ys);

      const gridPreds = tf.tidy(() => {
        const p = model.predict(gridTensor) as tf.Tensor2D;
        return Float32Array.from(p.dataSync());
      });

      drawMiniCanvas(ds, gridPreds);
      setTrainAcc(trEval.acc);
      setValAcc(vaEval.acc);
      setProgress(100);

      // Compute score out of 100
      const accOk = vaEval.acc >= mission.targetValAcc;
      const paramsOk = actualParams <= mission.maxParams;
      const quizOk = selectedAnswer === mission.correctIdx;

      const ptsAcc = accOk ? 50 : vaEval.acc >= mission.targetValAcc - 0.04 ? 25 : 0;
      const ptsParams = paramsOk && accOk ? 20 : 0;
      const ptsQuiz = quizOk ? 30 : 0;
      const totalPts = ptsAcc + ptsParams + ptsQuiz;
      const passed = totalPts >= 80;

      let feedback = mission.explanationFr;
      if (!accOk) {
        feedback = `Précision insuffisante (${(vaEval.acc * 100).toFixed(1)}% < ${Math.round(
          mission.targetValAcc * 100
        )}%). Ajoutez 1 couche cachée avec 6 à 8 neurones !`;
      } else if (!paramsOk) {
        feedback = `La précision est bonne, mais votre réseau est trop complexe (${actualParams} paramètres > max ${mission.maxParams}). Réduisez le nombre de couches !`;
      } else if (!quizOk) {
        feedback = `Le réseau performe très bien (${(vaEval.acc * 100).toFixed(
          1
        )}%), mais vérifiez votre réponse à la question théorique (Étape 2) pour avoir 100/100 !`;
      }

      setResults((prev) => ({
        ...prev,
        [mission.id]: {
          points: totalPts,
          passed,
          trainAcc: trEval.acc,
          valAcc: vaEval.acc,
          params: actualParams,
          feedback,
        },
      }));
    } finally {
      tr.xs.dispose();
      tr.ys.dispose();
      va.xs.dispose();
      va.ys.dispose();
      gridTensor.dispose();
      model.dispose();
      setIsRunning(false);
    }
  };

  const totalScore =
    (results[1]?.points || 0) + (results[2]?.points || 0) + (results[3]?.points || 0);
  const currentRes = results[mission.id];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Compact Header */}
      <div className="p-5 rounded-2xl bg-navy-900 border border-turquoise/30 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold uppercase text-turquoise">
            <Trophy className="w-4 h-4" />
            Quiz & Défi Pratique Rapide
          </div>
          <h2 className="text-xl font-extrabold text-white mt-0.5">
            Devenez l’architecte du réseau !
          </h2>
          <p className="text-xs text-slate-300 mt-0.5">
            Choisissez la bonne architecture, répondez à la question clé et validez en 1 clic.
          </p>
        </div>

        <div className="px-4 py-2.5 rounded-xl bg-navy-950 border border-turquoise/40 flex items-center gap-3">
          <Award className="w-6 h-6 text-turquoise" />
          <div>
            <div className="text-[10px] uppercase text-slate-400">Score Total</div>
            <div className="text-xl font-mono font-extrabold text-turquoise">
              {totalScore} <span className="text-xs text-slate-400">/ 300 pts</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3 Mission Pills */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {MISSIONS.map((m, idx) => {
          const r = results[m.id];
          const active = idx === missionIdx;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setMissionIdx(idx)}
              className={`p-3.5 rounded-xl border text-left transition-all flex items-center justify-between ${
                active
                  ? 'bg-navy-800 border-turquoise shadow-md'
                  : 'bg-navy-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div>
                <span className="text-[11px] font-bold text-turquoise uppercase">
                  Mission {m.id} • {m.badge}
                </span>
                <div className="text-xs font-bold text-white mt-0.5">
                  {m.title.split(':')[1]}
                </div>
              </div>
              {r && (
                <span
                  className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                    r.passed
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : 'bg-orange-500/20 text-orange-300'
                  }`}
                >
                  {r.points}/100
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Simple 2-Column Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-xl">
        {/* Left: Step 1 (Architecture) + Step 2 (Question) + Single Action Button */}
        <div className="lg:col-span-7 space-y-5">
          <div>
            <h3 className="text-base font-bold text-white">{mission.title}</h3>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              {mission.contextFr}
            </p>
          </div>

          {/* Step 1: Simple Architecture Selector */}
          <div className="p-4 rounded-xl bg-navy-950/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-turquoise uppercase flex items-center gap-1.5">
                <Layers className="w-4 h-4" />
                1. Choisissez l’architecture du réseau
              </span>
              <span
                className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                  paramsCount <= mission.maxParams
                    ? 'bg-emerald-500/15 text-emerald-300'
                    : 'bg-orange-500/20 text-orange-300'
                }`}
              >
                {paramsCount} paramètres (max conseillé : {mission.maxParams})
              </span>
            </div>

            {/* Quick button presets for Hidden Layers */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <div>
                <label className="text-[11px] text-slate-400 block mb-1">
                  Couches cachées
                </label>
                <div className="flex gap-1">
                  {[0, 1, 2, 4].map((l) => (
                    <button
                      key={l}
                      type="button"
                      onClick={() => setHiddenLayers(l)}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-mono font-bold border ${
                        hiddenLayers === l
                          ? 'bg-turquoise text-navy-950 border-turquoise'
                          : 'bg-navy-900 text-slate-300 border-slate-800'
                      }`}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">
                  Neurones par couche
                </label>
                <div className="flex gap-1">
                  {[4, 6, 8, 16].map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => {
                        if (hiddenLayers === 0) setHiddenLayers(1);
                        setNeurons(n);
                      }}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-mono font-bold border ${
                        hiddenLayers > 0 && neurons === n
                          ? 'bg-violet-500 text-white border-violet-400'
                          : 'bg-navy-900 text-slate-300 border-slate-800'
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">
                  Activation
                </label>
                <div className="flex gap-1">
                  {(['relu', 'tanh'] as ActivationType[]).map((act) => (
                    <button
                      key={act}
                      type="button"
                      onClick={() => {
                        if (hiddenLayers === 0) setHiddenLayers(1);
                        setActivation(act);
                      }}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-bold uppercase border ${
                        hiddenLayers > 0 && activation === act
                          ? 'bg-sky-500/30 text-sky-300 border-sky-400'
                          : 'bg-navy-900 text-slate-400 border-slate-800'
                      }`}
                    >
                      {act}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Step 2: Question */}
          <div className="p-4 rounded-xl bg-navy-950/90 border border-slate-800 space-y-2.5">
            <div className="text-xs font-bold text-violet-400 uppercase">
              2. Question de compréhension (30 pts)
            </div>
            <p className="text-xs font-semibold text-white">{mission.questionFr}</p>
            <div className="space-y-1.5">
              {mission.optionsFr.map((opt, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setSelectedAnswer(i)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs border transition-all ${
                    selectedAnswer === i
                      ? 'bg-violet-500/25 border-violet-400 text-white font-semibold'
                      : 'bg-navy-900/70 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  {String.fromCharCode(65 + i)}. {opt}
                </button>
              ))}
            </div>
          </div>

          {/* Single Action Button */}
          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={isRunning}
              onClick={handleRunAndEvaluate}
              className="flex-1 py-3 px-4 rounded-xl bg-turquoise hover:bg-turquoise-400 disabled:opacity-50 text-navy-950 font-extrabold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg transition-all"
            >
              <Play className="w-4 h-4 fill-current" />
              {isRunning
                ? `Test en cours (${progress}%)...`
                : 'Tester & Valider ma Mission'}
            </button>

            <button
              type="button"
              disabled={isRunning}
              onClick={() => {
                setHiddenLayers(mission.defaultLayers);
                setNeurons(mission.defaultNeurons);
                setSelectedAnswer(null);
                setTrainAcc(null);
                setValAcc(null);
                drawMiniCanvas(datasetRef.current);
              }}
              className="px-3.5 py-3 rounded-xl bg-navy-800 hover:bg-navy-700 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center gap-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>
        </div>

        {/* Right: Clean Visual Preview + Score Feedback */}
        <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
          <div className="p-4 rounded-xl bg-navy-950 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-300">Aperçu de la séparation</span>
              <span className="font-mono text-slate-400">
                Cible Validation ≥ {Math.round(mission.targetValAcc * 100)}%
              </span>
            </div>

            <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-navy-900 flex items-center justify-center h-48">
              <canvas
                ref={canvasRef}
                width={300}
                height={190}
                className="w-full h-full object-contain"
              />
            </div>

            {/* Simple 2-Metric Readout */}
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="p-2.5 rounded-lg bg-navy-900 border border-slate-800">
                <div className="text-[10px] uppercase text-slate-400">Train Accuracy</div>
                <div className="text-base font-mono font-bold text-sky-400">
                  {trainAcc !== null ? `${(trainAcc * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-navy-900 border border-slate-800">
                <div className="text-[10px] uppercase text-slate-400">
                  Validation Accuracy
                </div>
                <div
                  className={`text-base font-mono font-bold ${
                    valAcc === null
                      ? 'text-slate-400'
                      : valAcc >= mission.targetValAcc
                      ? 'text-turquoise'
                      : 'text-orange-400'
                  }`}
                >
                  {valAcc !== null ? `${(valAcc * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
            </div>
          </div>

          {/* Instant Score Box */}
          {currentRes ? (
            <div
              className={`p-4 rounded-xl border space-y-2 ${
                currentRes.passed
                  ? 'bg-emerald-950/40 border-emerald-400/60'
                  : 'bg-orange-950/40 border-orange-400/60'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-extrabold text-white flex items-center gap-1.5">
                  <CheckCircle2
                    className={`w-4 h-4 ${
                      currentRes.passed ? 'text-emerald-400' : 'text-orange-400'
                    }`}
                  />
                  Score : {currentRes.points} / 100 pts
                </span>
                {currentRes.passed && missionIdx < 2 && (
                  <button
                    type="button"
                    onClick={() => setMissionIdx((i) => i + 1)}
                    className="px-2.5 py-1 rounded-lg bg-turquoise text-navy-950 font-bold text-xs flex items-center gap-1"
                  >
                    Mission suivante <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-200 leading-relaxed">
                {currentRes.feedback}
              </p>
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-navy-950/60 border border-dashed border-slate-800 text-xs text-slate-400 text-center">
              Sélectionnez vos boutons à gauche puis cliquez sur{' '}
              <strong className="text-turquoise">Tester & Valider ma Mission</strong> pour voir votre résultat.
            </div>
          )}
        </div>
      </div>

      {/* Final Banner when all 3 are done */}
      {Object.keys(results).length === 3 && (
        <div className="p-5 rounded-2xl bg-gradient-to-r from-turquoise/20 via-navy-900 to-violet-600/20 border border-turquoise text-center space-y-2">
          <div className="inline-flex items-center gap-2 text-turquoise font-bold text-sm">
            <Sparkles className="w-4 h-4" />
            Bilan complété : {totalScore} / 300 points
          </div>
          <p className="text-xs text-slate-200">
            Vous avez terminé les 3 missions de synthèse du Neural Network Lab !
          </p>
        </div>
      )}
    </div>
  );
};
