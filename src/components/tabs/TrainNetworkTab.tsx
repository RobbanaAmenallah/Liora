import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  ActivationType,
  DatasetType,
  EpochMetric,
  GeneratedDataset,
  GuidedExperiment,
  LayerWeights,
  ModelComparisonSnapshot,
  ModelConfig,
} from '@/types/ml';
import { generateDataset } from '@/lib/datasets';
import {
  buildNormalizedGridTensor,
  countTrainableParameters,
  createNeuralNetwork,
  ensureTfBackend,
  evaluateSplit,
  extractModelWeights,
  pointsToTensors,
  predictDecisionGrid,
} from '@/lib/tfModel';
import { GUIDED_EXPERIMENTS } from '@/lib/educationalContent';
import { DecisionBoundaryCanvas } from '@/components/visualizations/DecisionBoundaryCanvas';
import { NetworkArchitectureSvg } from '@/components/visualizations/NetworkArchitectureSvg';
import { LearningCurvesChart } from '@/components/visualizations/LearningCurvesChart';
import { FrenchTooltip } from '@/components/ui/FrenchTooltip';
import {
  Play,
  Pause,
  RotateCcw,
  Cpu,
  GitCompare,
  Sparkles,
  Database,
  Layers,
  Activity,
  Trash2,
  AlertTriangle,
  Presentation,
  ChevronRight,
} from 'lucide-react';

interface TrainNetworkTabProps {
  presentationMode: boolean;
  activeExperimentId: 'A' | 'B' | 'C' | 'D' | null;
  onSelectExperiment: (expId: 'A' | 'B' | 'C' | 'D' | null) => void;
}

const GRID_RES = 34;

export const TrainNetworkTab: React.FC<TrainNetworkTabProps> = ({
  presentationMode,
  activeExperimentId,
  onSelectExperiment,
}) => {
  // Dataset settings (defaults per specification: Two Moons, 400 samples, reasonable noise 0.15, 80/20 split)
  const [datasetType, setDatasetType] = useState<DatasetType>('moons');
  const [numSamples, setNumSamples] = useState<number>(400);
  const [noise, setNoise] = useState<number>(0.15);
  const [trainSplit, setTrainSplit] = useState<number>(0.8);
  const [seed] = useState<number>(42);

  // Model Architecture & Hyperparameters
  const [config, setConfig] = useState<ModelConfig>({
    hiddenLayers: 1,
    neuronsPerLayer: 8,
    activation: 'relu',
    learningRate: 0.03,
    epochs: 80,
    batchSize: 32,
  });

  // Runtime state
  const [backendName, setBackendName] = useState<string>('cpu');
  const [dataset, setDataset] = useState<GeneratedDataset>(() =>
    generateDataset('moons', 400, 0.15, 0.8, 42)
  );
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [history, setHistory] = useState<EpochMetric[]>([]);
  const [trainableParams, setTrainableParams] = useState<number>(0);
  const [layerWeights, setLayerWeights] = useState<LayerWeights[]>([]);
  const [gridPredictions, setGridPredictions] = useState<Float32Array>(() =>
    new Float32Array(GRID_RES * GRID_RES).fill(0.5)
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Model comparison snapshots
  const [snapshots, setSnapshots] = useState<ModelComparisonSnapshot[]>([]);
  const [showComparePanel, setShowComparePanel] = useState<boolean>(false);

  // Guided Demonstration Mode interactive quiz state
  const [selectedPredictionOption, setSelectedPredictionOption] = useState<number | null>(
    null
  );

  // Mutable refs for safe TF.js disposal and async training loop interruption
  const modelRef = useRef<tf.Sequential | null>(null);
  const stopSignalRef = useRef<boolean>(false);
  const pauseSignalRef = useRef<boolean>(false);
  const currentEpochRef = useRef<number>(0);

  // Initialize TF backend once
  useEffect(() => {
    ensureTfBackend()
      .then((b) => setBackendName(b))
      .catch((err) => setErrorMessage(`Erreur d'initialisation TensorFlow.js: ${String(err)}`));
  }, []);

  /**
   * Disposes previous model safely and creates a fresh TensorFlow.js model
   * synchronized with current dataset and config.
   */
  const initializeModel = useCallback(
    (
      targetDataset: GeneratedDataset,
      targetConfig: ModelConfig,
      modelSeed: number
    ) => {
      stopSignalRef.current = true;
      pauseSignalRef.current = false;
      setIsTraining(false);
      setIsPaused(false);
      setErrorMessage(null);

      try {
        if (modelRef.current) {
          modelRef.current.dispose();
          modelRef.current = null;
        }

        const newModel = createNeuralNetwork(targetConfig, modelSeed);
        modelRef.current = newModel;

        const paramsCount = countTrainableParameters(newModel);
        setTrainableParams(paramsCount);
        setLayerWeights(extractModelWeights(newModel));
        setCurrentEpoch(0);
        currentEpochRef.current = 0;

        // Evaluate initial Epoch 0 metrics on train and val splits
        const trainTensors = pointsToTensors(
          targetDataset.trainPoints,
          targetDataset.stats
        );
        const valTensors = pointsToTensors(targetDataset.valPoints, targetDataset.stats);

        const initTrain = evaluateSplit(newModel, trainTensors.xs, trainTensors.ys);
        const initVal = evaluateSplit(newModel, valTensors.xs, valTensors.ys);

        trainTensors.xs.dispose();
        trainTensors.ys.dispose();
        valTensors.xs.dispose();
        valTensors.ys.dispose();

        setHistory([
          {
            epoch: 0,
            loss: initTrain.loss,
            acc: initTrain.acc,
            valLoss: initVal.loss,
            valAcc: initVal.acc,
          },
        ]);

        const preds = predictDecisionGrid(
          newModel,
          targetDataset.bounds,
          targetDataset.stats,
          GRID_RES
        );
        setGridPredictions(preds);
      } catch (err) {
        console.error(err);
        setErrorMessage(`Erreur lors de la construction du modèle : ${String(err)}`);
      }
    },
    []
  );

  // Single unified effect to rebuild dataset (when dataset params change) and initialize model once
  const prevDsParamsRef = useRef({
    datasetType,
    numSamples,
    noise,
    trainSplit,
    seed,
  });

  useEffect(() => {
    const prev = prevDsParamsRef.current;
    const dsChanged =
      prev.datasetType !== datasetType ||
      prev.numSamples !== numSamples ||
      prev.noise !== noise ||
      prev.trainSplit !== trainSplit ||
      prev.seed !== seed;

    let activeDs = dataset;
    if (dsChanged) {
      prevDsParamsRef.current = { datasetType, numSamples, noise, trainSplit, seed };
      activeDs = generateDataset(datasetType, numSamples, noise, trainSplit, seed);
      setDataset(activeDs);
    }

    initializeModel(activeDs, config, seed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    datasetType,
    numSamples,
    noise,
    trainSplit,
    seed,
    config.hiddenLayers,
    config.neuronsPerLayer,
    config.activation,
    config.learningRate,
    initializeModel,
  ]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSignalRef.current = true;
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    };
  }, []);

  // Apply guided experiment preset when selected
  useEffect(() => {
    if (!activeExperimentId) return;
    const exp: GuidedExperiment | undefined = GUIDED_EXPERIMENTS.find(
      (e) => e.id === activeExperimentId
    );
    if (!exp) return;

    stopSignalRef.current = true;
    setSelectedPredictionOption(null);
    setDatasetType(exp.dataset);
    setNumSamples(exp.samples);
    setNoise(exp.noise);
    setTrainSplit(exp.trainSplit);
    setConfig(exp.config);
  }, [activeExperimentId]);

  /**
   * Genuine TensorFlow.js training loop with throttled UI updates (~20fps max)
   * and pre-allocated grid tensor for ultra-smooth zero-lag rendering.
   */
  const runTrainingLoop = async (startFromEpoch: number) => {
    if (!modelRef.current) {
      initializeModel(dataset, config, seed);
    }
    const model = modelRef.current;
    if (!model) return;

    stopSignalRef.current = false;
    pauseSignalRef.current = false;
    setIsTraining(true);
    setIsPaused(false);
    setErrorMessage(null);

    const trainTensors = pointsToTensors(dataset.trainPoints, dataset.stats);
    const valTensors = pointsToTensors(dataset.valPoints, dataset.stats);
    const cachedGridTensor = buildNormalizedGridTensor(
      dataset.bounds,
      dataset.stats,
      GRID_RES
    );

    const pendingMetrics: EpochMetric[] = [];
    let lastUiFlushTime = performance.now();

    try {
      for (let ep = startFromEpoch + 1; ep <= config.epochs; ep++) {
        if (stopSignalRef.current || pauseSignalRef.current) {
          break;
        }

        // Fit 1 epoch strictly on training tensors
        await model.fit(trainTensors.xs, trainTensors.ys, {
          epochs: 1,
          batchSize: config.batchSize,
          shuffle: true,
          verbose: 0,
        });

        if (stopSignalRef.current) break;

        // Fast typed-array evaluation on train & val splits
        const trainEval = evaluateSplit(model, trainTensors.xs, trainTensors.ys);
        const valEval = evaluateSplit(model, valTensors.xs, valTensors.ys);

        currentEpochRef.current = ep;
        pendingMetrics.push({
          epoch: ep,
          loss: trainEval.loss,
          acc: trainEval.acc,
          valLoss: valEval.loss,
          valAcc: valEval.acc,
        });

        const now = performance.now();
        const isLastEpoch = ep === config.epochs;
        const shouldFlushUi =
          isLastEpoch ||
          pauseSignalRef.current ||
          (ep % 4 === 0 && now - lastUiFlushTime >= 55);

        if (shouldFlushUi) {
          const batchToAppend = [...pendingMetrics];
          pendingMetrics.length = 0;

          const grid = predictDecisionGrid(
            model,
            dataset.bounds,
            dataset.stats,
            GRID_RES,
            cachedGridTensor
          );
          const weights = extractModelWeights(model);

          setCurrentEpoch(ep);
          setHistory((prev) => [...prev, ...batchToAppend]);
          setGridPredictions(grid);
          setLayerWeights(weights);

          lastUiFlushTime = performance.now();
          // Yield to browser animation frame only when flushing UI
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        }
      }

      // Flush any remaining metrics if paused/stopped mid-batch
      if (pendingMetrics.length > 0 && !stopSignalRef.current) {
        const remaining = [...pendingMetrics];
        setCurrentEpoch(currentEpochRef.current);
        setHistory((prev) => [...prev, ...remaining]);
        setGridPredictions(
          predictDecisionGrid(
            model,
            dataset.bounds,
            dataset.stats,
            GRID_RES,
            cachedGridTensor
          )
        );
        setLayerWeights(extractModelWeights(model));
      }
    } catch (err) {
      console.error('Training error:', err);
      setErrorMessage(`Erreur pendant l'entraînement TensorFlow.js : ${String(err)}`);
    } finally {
      trainTensors.xs.dispose();
      trainTensors.ys.dispose();
      valTensors.xs.dispose();
      valTensors.ys.dispose();
      cachedGridTensor.dispose();

      setIsTraining(false);
    }
  };

  const handleStartOrRestartTrain = () => {
    if (currentEpochRef.current >= config.epochs) {
      initializeModel(dataset, config, seed);
      setTimeout(() => runTrainingLoop(0), 16);
    } else {
      runTrainingLoop(currentEpochRef.current);
    }
  };

  const handlePauseResume = () => {
    if (isTraining) {
      pauseSignalRef.current = true;
      setIsPaused(true);
      setIsTraining(false);
    } else if (isPaused) {
      runTrainingLoop(currentEpochRef.current);
    }
  };

  const handleResetAll = () => {
    stopSignalRef.current = true;
    pauseSignalRef.current = false;
    initializeModel(dataset, config, seed);
  };

  const handleSaveSnapshot = () => {
    const latest = history[history.length - 1];
    if (!latest) return;

    const archDesc =
      config.hiddenLayers === 0
        ? 'Linéaire (0 couche cachée)'
        : `${config.hiddenLayers}×${config.neuronsPerLayer} (${config.activation.toUpperCase()})`;

    const newSnap: ModelComparisonSnapshot = {
      id: `${Date.now()}`,
      name: `Modèle #${snapshots.length + 1} — ${archDesc}`,
      timestamp: new Date().toLocaleTimeString('fr-FR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }),
      datasetType,
      noise,
      config: { ...config },
      trainableParams,
      finalEpoch: currentEpoch,
      trainLoss: latest.loss,
      valLoss: latest.valLoss,
      trainAcc: latest.acc,
      valAcc: latest.valAcc,
      history: [...history],
    };

    setSnapshots((prev) => [newSnap, ...prev]);
    setShowComparePanel(true);
  };

  const latestMetric = history[history.length - 1] || {
    epoch: 0,
    loss: 0.693,
    acc: 0.5,
    valLoss: 0.693,
    valAcc: 0.5,
  };

  const activeExperiment = GUIDED_EXPERIMENTS.find((e) => e.id === activeExperimentId);

  return (
    <div className="space-y-5">
      {/* Guided Demonstration Banner */}
      <div className="p-4 rounded-2xl bg-gradient-to-r from-navy-800 via-navy-800/95 to-violet-950/60 border border-turquoise/30 shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-xl bg-turquoise/15 text-turquoise">
              <Presentation className="w-5 h-5" />
            </span>
            <div>
              <h2 className="text-sm md:text-base font-bold text-white flex items-center gap-2">
                Mode Démonstration Guidée (4 Expériences Masterclass)
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-turquoise/15 text-turquoise font-mono">
                  Backend TF.js : {backendName.toUpperCase()} (Ultra-Fast)
                </span>
              </h2>
              <p className="text-xs text-slate-300">
                Sélectionnez une expérience ci-dessous ou cliquez sur « Démarrer la démonstration » dans l’en-tête pour guider les participants.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {GUIDED_EXPERIMENTS.map((exp) => {
              const isSelected = activeExperimentId === exp.id;
              return (
                <button
                  key={exp.id}
                  type="button"
                  onClick={() => onSelectExperiment(isSelected ? null : exp.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all border ${
                    isSelected
                      ? 'bg-turquoise text-navy-950 border-turquoise shadow-[0_0_15px_rgba(89,217,197,0.4)]'
                      : 'bg-navy-900/90 text-slate-200 border-slate-700 hover:border-turquoise/50'
                  }`}
                >
                  Expérience {exp.id}
                </button>
              );
            })}
          </div>
        </div>

        {/* Expanded Pedagogical Panel when an Experiment is selected */}
        {activeExperiment && (
          <div className="mt-4 pt-4 border-t border-slate-700/80 grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-6 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold uppercase tracking-wider text-turquoise">
                  {activeExperiment.title}
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  {activeExperiment.config.hiddenLayers} couche(s) cachée(s) • Bruit {activeExperiment.noise}
                </span>
              </div>
              <p className="text-xs md:text-sm text-slate-200 leading-relaxed">
                {activeExperiment.explanationFr}
              </p>
              <div className="p-2.5 rounded-xl bg-navy-950/80 border border-slate-800 text-xs text-violet-300">
                <strong>Message clé :</strong> {activeExperiment.takeawayFr}
              </div>
            </div>

            <div className="lg:col-span-6 p-3.5 rounded-xl bg-navy-950/90 border border-violet-500/40 space-y-2.5">
              <div className="text-xs font-bold text-violet-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-turquoise" />
                Question à poser à la classe avant d’entraîner :
              </div>
              <p className="text-xs font-semibold text-white">
                {activeExperiment.questionFr}
              </p>
              <div className="space-y-1.5">
                {activeExperiment.optionsFr.map((opt, idx) => {
                  const chosen = selectedPredictionOption === idx;
                  const isCorrect = idx === activeExperiment.correctOptionIndex;
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setSelectedPredictionOption(idx)}
                      className={`w-full text-left px-3 py-1.5 rounded-lg text-xs border transition-all flex items-center justify-between ${
                        chosen
                          ? isCorrect
                            ? 'bg-emerald-500/20 border-emerald-400 text-emerald-200 font-semibold'
                            : 'bg-orange-500/20 border-orange-400 text-orange-200'
                          : 'bg-navy-900/70 border-slate-800 text-slate-300 hover:border-slate-600'
                      }`}
                    >
                      <span>
                        {String.fromCharCode(65 + idx)}. {opt}
                      </span>
                      {chosen && (
                        <span className="text-[11px] font-bold ml-2 shrink-0">
                          {isCorrect ? '✓ Bonne prédiction' : '✗ Testez avec Train !'}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
              <div className="flex items-center justify-between pt-1">
                <button
                  type="button"
                  onClick={handleStartOrRestartTrain}
                  disabled={isTraining}
                  className="px-3.5 py-1.5 rounded-lg bg-turquoise hover:bg-turquoise-400 text-navy-950 font-bold text-xs flex items-center gap-1.5 shadow-md"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Lancer l’entraînement de l’Expérience {activeExperiment.id}
                </button>
                {activeExperiment.id !== 'D' && (
                  <button
                    type="button"
                    onClick={() => {
                      const nextMap: Record<'A' | 'B' | 'C' | 'D', 'B' | 'C' | 'D'> = {
                        A: 'B',
                        B: 'C',
                        C: 'D',
                        D: 'D',
                      };
                      onSelectExperiment(nextMap[activeExperiment.id]);
                    }}
                    className="text-xs text-turquoise hover:underline flex items-center gap-1"
                  >
                    Expérience suivante <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {errorMessage && (
        <div className="p-3 rounded-xl bg-red-500/20 border border-red-400 text-xs text-red-200 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Action Bar & Live Training Metrics Strip */}
      <div className="p-4 rounded-2xl bg-navy-900/95 border border-slate-800 shadow-lg flex flex-wrap items-center justify-between gap-4">
        {/* Training Control Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => initializeModel(dataset, config, seed)}
            disabled={isTraining}
            className="px-3.5 py-2 rounded-xl bg-navy-800 hover:bg-navy-700 disabled:opacity-50 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-1.5 transition-all"
          >
            <Cpu className="w-4 h-4 text-sky-400" />
            Initialize Model
          </button>

          <button
            type="button"
            onClick={handleStartOrRestartTrain}
            disabled={isTraining}
            className="px-4 py-2 rounded-xl bg-turquoise hover:bg-turquoise-400 disabled:opacity-50 text-navy-950 font-extrabold text-xs flex items-center gap-1.5 shadow-[0_0_20px_rgba(89,217,197,0.35)] transition-all"
          >
            <Play className="w-4 h-4 fill-current" />
            {currentEpoch > 0 && currentEpoch < config.epochs ? 'Continuer Train' : 'Train'}
          </button>

          <button
            type="button"
            onClick={handlePauseResume}
            disabled={!isTraining && !isPaused}
            className="px-3.5 py-2 rounded-xl bg-violet-600/30 hover:bg-violet-600/50 disabled:opacity-40 border border-violet-400/40 text-xs font-semibold text-violet-200 flex items-center gap-1.5 transition-all"
          >
            <Pause className="w-4 h-4" />
            {isPaused ? 'Resume' : 'Pause'}
          </button>

          <button
            type="button"
            onClick={handleResetAll}
            className="px-3.5 py-2 rounded-xl bg-navy-800 hover:bg-navy-700 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center gap-1.5 transition-all"
          >
            <RotateCcw className="w-4 h-4 text-orange-400" />
            Reset
          </button>

          <button
            type="button"
            onClick={handleSaveSnapshot}
            className="px-3.5 py-2 rounded-xl bg-navy-800 hover:bg-navy-700 border border-turquoise/40 text-xs font-semibold text-turquoise flex items-center gap-1.5 transition-all"
          >
            <GitCompare className="w-4 h-4" />
            Compare Models ({snapshots.length})
          </button>
        </div>

        {/* Live Metrics Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 w-full xl:w-auto">
          <div className="px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Epoch</div>
            <div className="text-sm font-mono font-bold text-white">
              {currentEpoch} <span className="text-slate-500 text-xs">/ {config.epochs}</span>
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Train Acc</div>
            <div className="text-sm font-mono font-bold text-sky-400">
              {(latestMetric.acc * 100).toFixed(1)}%
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Val Acc</div>
            <div className="text-sm font-mono font-bold text-turquoise">
              {(latestMetric.valAcc * 100).toFixed(1)}%
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Current Loss</div>
            <div className="text-sm font-mono font-bold text-orange-400">
              {latestMetric.loss.toFixed(4)}
            </div>
          </div>

          <div className="px-3 py-1.5 rounded-xl bg-navy-950 border border-slate-800">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">Paramètres</div>
            <div className="text-sm font-mono font-bold text-violet-400">
              {trainableParams}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Layout: Left Config Sidebar (hidden in Presentation Mode) + Right Visualizations */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {!presentationMode && (
          <div className="xl:col-span-3 space-y-4">
            {/* Dataset Generator Panel */}
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 space-y-3.5">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-turquoise flex items-center gap-1.5">
                  <Database className="w-4 h-4" />
                  Jeu de Données (Dataset)
                </h3>
                <FrenchTooltip
                  term="Dataset Generator"
                  explanation="Génère des points 2D synthétiques avec graine aléatoire fixe (reproductible). Les statistiques de standardisation (moyenne/écart-type) sont calculées exclusivement sur le jeu d'entraînement."
                  sklearnNote="Équivalent de make_moons, make_circles + StandardScaler().fit_transform(X_train)."
                />
              </div>

              <div className="grid grid-cols-2 gap-1.5">
                {(
                  [
                    { id: 'moons', label: 'Two Moons' },
                    { id: 'circles', label: 'Circles' },
                    { id: 'xor', label: 'XOR' },
                    { id: 'linear', label: 'Linéaire' },
                  ] as const
                ).map((ds) => (
                  <button
                    key={ds.id}
                    type="button"
                    disabled={isTraining}
                    onClick={() => {
                      onSelectExperiment(null);
                      setDatasetType(ds.id);
                    }}
                    className={`px-2.5 py-2 rounded-xl text-xs font-bold border transition-all ${
                      datasetType === ds.id
                        ? 'bg-turquoise/20 border-turquoise text-turquoise shadow-sm'
                        : 'bg-navy-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {ds.label}
                  </button>
                ))}
              </div>

              {/* Samples slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Échantillons (Samples)
                    <FrenchTooltip
                      term="Nombre d'échantillons"
                      explanation="Taille totale du dataset généré (divisé ensuite entre Train et Validation de manière stratifiée)."
                    />
                  </span>
                  <span className="font-mono font-bold text-turquoise">{numSamples}</span>
                </div>
                <input
                  type="range"
                  min={100}
                  max={800}
                  step={50}
                  disabled={isTraining}
                  value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value, 10))}
                  className="w-full"
                />
              </div>

              {/* Noise slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Bruit gaussien (Noise)
                    <FrenchTooltip
                      term="Bruit (Noise)"
                      explanation="Écart-type du bruit aléatoire ajouté aux coordonnées. Un bruit élevé fait se chevaucher les classes et favorise l'Overfitting si le réseau est trop complexe."
                    />
                  </span>
                  <span className="font-mono font-bold text-orange-400">
                    {noise.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.0}
                  max={0.45}
                  step={0.05}
                  disabled={isTraining}
                  value={noise}
                  onChange={(e) => setNoise(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Train/Val split selector */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Split Train / Validation
                    <FrenchTooltip
                      term="Train / Validation Split"
                      explanation="Répartition stratifiée. Le modèle n'apprend JAMAIS sur les points de validation : ils servent uniquement à mesurer la capacité de généralisation."
                      sklearnNote="train_test_split(X, y, train_size=0.8, stratify=y)"
                    />
                  </span>
                  <span className="font-mono font-bold text-sky-400">
                    {Math.round(trainSplit * 100)}% / {Math.round((1 - trainSplit) * 100)}%
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1.5 pt-1">
                  {[0.7, 0.8, 0.9].map((ratio) => (
                    <button
                      key={ratio}
                      type="button"
                      disabled={isTraining}
                      onClick={() => setTrainSplit(ratio)}
                      className={`py-1 rounded-lg text-xs font-mono border ${
                        Math.abs(trainSplit - ratio) < 0.01
                          ? 'bg-sky-500/20 border-sky-400 text-sky-300 font-bold'
                          : 'bg-navy-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      {Math.round(ratio * 100)}/{Math.round((1 - ratio) * 100)}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Architecture & Hyperparameters Panel */}
            <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 space-y-3.5">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-violet-400 flex items-center gap-1.5">
                  <Layers className="w-4 h-4" />
                  Architecture & Hyperparamètres
                </h3>
                <FrenchTooltip
                  term="Architecture du réseau"
                  explanation="Configurez le nombre de couches cachées (0 = régression logistique linéaire) et de neurones par couche."
                />
              </div>

              {/* Hidden layers: 0 to 4 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Couches cachées (Hidden Layers)
                    <FrenchTooltip
                      term="Hidden Layers (0 à 4)"
                      explanation="À 0 couche cachée, le modèle est un classifieur linéaire (LogisticRegression). Dès 1 couche cachée, il devient un perceptron multicouche (MLP)."
                    />
                  </span>
                  <span className="font-mono font-bold text-violet-400">
                    {config.hiddenLayers}{' '}
                    {config.hiddenLayers === 0 ? '(Linéaire)' : ''}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={4}
                  step={1}
                  disabled={isTraining}
                  value={config.hiddenLayers}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      hiddenLayers: parseInt(e.target.value, 10),
                    }))
                  }
                  className="w-full"
                />
              </div>

              {/* Neurons per layer: 1 to 16 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Neurones par couche
                    <FrenchTooltip
                      term="Neurons per Hidden Layer (1 à 16)"
                      explanation="Largeur de chaque couche cachée. Plus il y a de neurones, plus la frontière peut comporter de segments ou de courbures complexes."
                    />
                  </span>
                  <span className="font-mono font-bold text-turquoise">
                    {config.hiddenLayers === 0 ? 'N/A' : config.neuronsPerLayer}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={16}
                  step={1}
                  disabled={isTraining || config.hiddenLayers === 0}
                  value={config.neuronsPerLayer}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      neuronsPerLayer: parseInt(e.target.value, 10),
                    }))
                  }
                  className="w-full disabled:opacity-30"
                />
              </div>

              {/* Activation selector */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Activation cachée
                    <FrenchTooltip
                      term="Fonction d'activation"
                      explanation="Fonction non-linéaire appliquée à chaque couche cachée. La couche de sortie utilise toujours Sigmoid pour produire P(y=1) ∈ [0, 1]."
                    />
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-1.5 pt-1">
                  {(['relu', 'tanh', 'sigmoid'] as ActivationType[]).map((act) => (
                    <button
                      key={act}
                      type="button"
                      disabled={isTraining || config.hiddenLayers === 0}
                      onClick={() => setConfig((c) => ({ ...c, activation: act }))}
                      className={`py-1.5 rounded-lg text-xs font-bold uppercase border transition-all disabled:opacity-40 ${
                        config.activation === act
                          ? 'bg-violet-500/25 border-violet-400 text-white'
                          : 'bg-navy-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      {act}
                    </button>
                  ))}
                </div>
              </div>

              {/* Learning Rate */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 flex items-center">
                    Learning Rate (Adam)
                    <FrenchTooltip
                      term="Learning Rate"
                      explanation="Taille du pas lors de la descente de gradient (optimiseur Adam)."
                    />
                  </span>
                  <span className="font-mono font-bold text-turquoise">
                    {config.learningRate}
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-1 pt-1">
                  {[0.003, 0.01, 0.03, 0.1].map((lr) => (
                    <button
                      key={lr}
                      type="button"
                      disabled={isTraining}
                      onClick={() => setConfig((c) => ({ ...c, learningRate: lr }))}
                      className={`py-1 rounded-lg text-[11px] font-mono border ${
                        config.learningRate === lr
                          ? 'bg-turquoise/20 border-turquoise text-turquoise font-bold'
                          : 'bg-navy-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      {lr}
                    </button>
                  ))}
                </div>
              </div>

              {/* Epochs & Batch Size */}
              <div className="grid grid-cols-2 gap-2.5 pt-1">
                <div>
                  <label className="text-[11px] text-slate-400 flex items-center mb-1">
                    Epochs
                    <FrenchTooltip
                      term="Epochs"
                      explanation="Nombre de passages complets sur l'ensemble des données d'entraînement."
                    />
                  </label>
                  <select
                    disabled={isTraining}
                    value={config.epochs}
                    onChange={(e) =>
                      setConfig((c) => ({ ...c, epochs: parseInt(e.target.value, 10) }))
                    }
                    className="w-full px-2.5 py-1.5 rounded-lg bg-navy-950 border border-slate-800 text-xs font-mono text-white"
                  >
                    {[40, 60, 80, 120, 200].map((ep) => (
                      <option key={ep} value={ep}>
                        {ep} epochs
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-[11px] text-slate-400 flex items-center mb-1">
                    Batch Size
                    <FrenchTooltip
                      term="Batch Size"
                      explanation="Nombre d'exemples utilisés pour calculer chaque mise à jour du gradient."
                    />
                  </label>
                  <select
                    disabled={isTraining}
                    value={config.batchSize}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        batchSize: parseInt(e.target.value, 10),
                      }))
                    }
                    className="w-full px-2.5 py-1.5 rounded-lg bg-navy-950 border border-slate-800 text-xs font-mono text-white"
                  >
                    {[16, 32, 64, 128].map((bs) => (
                      <option key={bs} value={bs}>
                        {bs} samples
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Right Visualizations Area */}
        <div
          className={
            presentationMode ? 'xl:col-span-12 space-y-5' : 'xl:col-span-9 space-y-5'
          }
        >
          {/* Top Row: A. Decision Boundary + B. Neural Network Architecture */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            {/* A. Decision Boundary Panel */}
            <div className="lg:col-span-6 p-4 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-lg flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-turquoise" />
                    A. Frontière de Décision (Decision Boundary 2D)
                  </h3>
                  <FrenchTooltip
                    term="Decision Boundary & Heatmap"
                    explanation="Carte de probabilité P(y=1) prédite par le modèle TensorFlow.js sur une grille normalisée. La courbe turquoise indique la frontière exacte P(y=1) = 0.5."
                  />
                </div>
                <span className="text-xs font-mono text-slate-400">
                  {dataset.trainPoints.length} train / {dataset.valPoints.length} val
                </span>
              </div>

              <div className="flex-1">
                <DecisionBoundaryCanvas
                  points={dataset.points}
                  bounds={dataset.bounds}
                  stats={dataset.stats}
                  gridPredictions={gridPredictions}
                  gridResolution={GRID_RES}
                  presentationMode={presentationMode}
                />
              </div>
            </div>

            {/* B. Neural Network Architecture Panel */}
            <div className="lg:col-span-6 p-4 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-lg flex flex-col">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <Layers className="w-4 h-4 text-violet-400" />
                    B. Architecture & Poids Synaptiques Appris
                  </h3>
                  <FrenchTooltip
                    term="Visualisation des Poids (Weights)"
                    explanation="Affiche la structure réelle du réseau et la valeur actuelle des poids appris. L'épaisseur reflète |w| et la couleur indique le signe (+ turquoise, - orange)."
                  />
                </div>
                <span className="text-xs font-mono text-violet-300">
                  {trainableParams} paramètres entraînables
                </span>
              </div>

              <div className="flex-1">
                <NetworkArchitectureSvg
                  config={config}
                  layerWeights={layerWeights}
                  presentationMode={presentationMode}
                />
              </div>
            </div>
          </div>

          {/* C. Learning Curves Panel */}
          <LearningCurvesChart
            history={history}
            presentationMode={presentationMode}
          />

          {/* Model Comparison Panel */}
          {(showComparePanel || snapshots.length > 0) && (
            <div className="p-4 rounded-2xl bg-navy-900/95 border border-turquoise/40 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <GitCompare className="w-4 h-4 text-turquoise" />
                  <h3 className="text-sm font-bold text-white">
                    Tableau Comparatif des Modèles Enregistrés
                  </h3>
                  <span className="text-xs text-slate-400">
                    (Cliquez sur « Compare Models » après un entraînement pour ajouter une ligne)
                  </span>
                </div>
                {snapshots.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSnapshots([])}
                    className="text-xs text-slate-400 hover:text-red-400 flex items-center gap-1"
                  >
                    <Trash2 className="w-3.5 h-3.5" /> Vider
                  </button>
                )}
              </div>

              {snapshots.length === 0 ? (
                <p className="text-xs text-slate-400">
                  Aucun modèle enregistré pour le moment. Entraînez une architecture puis cliquez sur le bouton{' '}
                  <strong className="text-turquoise">Compare Models</strong> pour comparer ses performances.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                        <th className="py-2 px-3">Modèle / Architecture</th>
                        <th className="py-2 px-3">Dataset (Bruit)</th>
                        <th className="py-2 px-3">Paramètres</th>
                        <th className="py-2 px-3">Epochs</th>
                        <th className="py-2 px-3">Train Acc</th>
                        <th className="py-2 px-3">Val Acc</th>
                        <th className="py-2 px-3">Train Loss</th>
                        <th className="py-2 px-3">Val Loss</th>
                        <th className="py-2 px-3">Diagnostic</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/70 font-mono">
                      {snapshots.map((snap) => {
                        const gap = snap.trainAcc - snap.valAcc;
                        let diag = 'Équilibré';
                        let diagColor = 'text-turquoise';
                        if (snap.valAcc < 0.82 && snap.config.hiddenLayers === 0) {
                          diag = 'Underfitting (Linéaire)';
                          diagColor = 'text-orange-400';
                        } else if (gap > 0.08 || snap.valLoss > snap.trainLoss * 1.35) {
                          diag = 'Risque Overfitting';
                          diagColor = 'text-orange-400';
                        }

                        return (
                          <tr key={snap.id} className="hover:bg-navy-800/50">
                            <td className="py-2.5 px-3 font-sans font-semibold text-white">
                              {snap.name}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {snap.datasetType} ({snap.noise.toFixed(2)})
                            </td>
                            <td className="py-2.5 px-3 text-violet-300 font-bold">
                              {snap.trainableParams}
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">{snap.finalEpoch}</td>
                            <td className="py-2.5 px-3 text-sky-400 font-bold">
                              {(snap.trainAcc * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 px-3 text-turquoise font-bold">
                              {(snap.valAcc * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 px-3 text-slate-300">
                              {snap.trainLoss.toFixed(4)}
                            </td>
                            <td className="py-2.5 px-3 text-orange-300">
                              {snap.valLoss.toFixed(4)}
                            </td>
                            <td className={`py-2.5 px-3 font-sans font-bold ${diagColor}`}>
                              {diag}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
