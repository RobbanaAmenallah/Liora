import * as tf from '@tensorflow/tfjs';
import {
  DataPoint,
  LayerWeights,
  ModelConfig,
  NormalizationStats,
} from '@/types/ml';
import { normalizePoint } from './datasets';

let backendInitialized = false;
let modelInstanceCounter = 0;

/**
 * Initialize TensorFlow.js.
 * For small 2D educational neural networks (400 samples, 1-4 layers of 1-16 neurons),
 * the CPU backend is 10x to 30x faster and far smoother than WebGL because WebGL incurs
 * synchronous GPU-to-CPU shader compilation & texture readback stalls on every batch!
 */
export async function ensureTfBackend(): Promise<string> {
  if (backendInitialized) {
    return tf.getBackend() || 'cpu';
  }
  try {
    await tf.setBackend('cpu');
    await tf.ready();
    backendInitialized = true;
    return tf.getBackend() || 'cpu';
  } catch (err) {
    console.warn('Fallback initialization:', err);
    await tf.ready();
    backendInitialized = true;
    return tf.getBackend() || 'cpu';
  }
}

/**
 * Builds and compiles a genuine TensorFlow.js Sequential neural network.
 * When hiddenLayers === 0, creates a single logistic regression dense layer (Sigmoid).
 */
export function createNeuralNetwork(
  config: ModelConfig,
  seed: number = 42
): tf.Sequential {
  modelInstanceCounter += 1;
  const uid = modelInstanceCounter;
  const model = tf.sequential();

  if (config.hiddenLayers === 0) {
    model.add(
      tf.layers.dense({
        inputShape: [2],
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: tf.initializers.glorotNormal({ seed }),
        biasInitializer: tf.initializers.zeros(),
        name: `sortie_logistique_${uid}`,
      })
    );
  } else {
    for (let i = 0; i < config.hiddenLayers; i++) {
      const initializer =
        config.activation === 'relu'
          ? tf.initializers.heNormal({ seed: seed + i * 17 })
          : tf.initializers.glorotNormal({ seed: seed + i * 17 });

      model.add(
        tf.layers.dense({
          inputShape: i === 0 ? [2] : undefined,
          units: config.neuronsPerLayer,
          activation: config.activation,
          kernelInitializer: initializer,
          biasInitializer: tf.initializers.zeros(),
          name: `couche_cachee_${uid}_${i + 1}`,
        })
      );
    }

    model.add(
      tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: tf.initializers.glorotNormal({ seed: seed + 99 }),
        biasInitializer: tf.initializers.zeros(),
        name: `couche_sortie_${uid}`,
      })
    );
  }

  const optimizer = tf.train.adam(config.learningRate);
  model.compile({
    optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

/**
 * Counts the exact number of trainable parameters in a compiled model.
 */
export function countTrainableParameters(model: tf.Sequential | null): number {
  if (!model) return 0;
  let total = 0;
  for (const weight of model.trainableWeights) {
    const shape = weight.shape;
    const size = shape.reduce<number>((acc, dim) => acc * (dim ?? 1), 1);
    total += size;
  }
  return total;
}

/**
 * Extracts weights and biases from every layer for interactive architecture visualization.
 */
export function extractModelWeights(model: tf.Sequential | null): LayerWeights[] {
  if (!model) return [];
  const result: LayerWeights[] = [];

  model.layers.forEach((layer, idx) => {
    const weights = layer.getWeights();
    if (weights.length >= 2) {
      const kernelTensor = weights[0];
      const biasTensor = weights[1];

      const shape = kernelTensor.shape;
      const inputSize = shape[0] || 1;
      const outputSize = shape[1] || 1;

      const kernelFlat = kernelTensor.dataSync();
      const biasFlat = biasTensor.dataSync();

      const matrix: number[][] = [];
      for (let i = 0; i < inputSize; i++) {
        const row: number[] = [];
        for (let j = 0; j < outputSize; j++) {
          row.push(kernelFlat[i * outputSize + j]);
        }
        matrix.push(row);
      }

      result.push({
        layerIndex: idx,
        layerName: layer.name,
        inputSize,
        outputSize,
        weights: matrix,
        biases: Array.from(biasFlat),
      });
    }
  });

  return result;
}

/**
 * Converts DataPoint[] into normalized TF tensors X and y.
 * Uses statistics computed ONLY on the training set.
 */
export function pointsToTensors(
  points: DataPoint[],
  stats: NormalizationStats
): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const n = points.length;
  const flatX = new Float32Array(n * 2);
  const flatY = new Float32Array(n);

  const invStd1 = 1 / (stats.stdX1 || 1);
  const invStd2 = 1 / (stats.stdX2 || 1);

  for (let i = 0; i < n; i++) {
    const p = points[i];
    flatX[i * 2] = (p.x1 - stats.meanX1) * invStd1;
    flatX[i * 2 + 1] = (p.x2 - stats.meanX2) * invStd2;
    flatY[i] = p.label;
  }

  return {
    xs: tf.tensor2d(flatX, [n, 2]),
    ys: tf.tensor2d(flatY, [n, 1]),
  };
}

/**
 * Fast, zero-overhead evaluation of Binary Cross-Entropy loss and Accuracy.
 * Performs a single model.predict(xs) call and computes BCE + accuracy in a tight
 * typed-array loop to avoid creating 12 intermediate TF tensors per epoch.
 */
export function evaluateSplit(
  model: tf.Sequential,
  xs: tf.Tensor2D,
  ys: tf.Tensor2D
): { loss: number; acc: number } {
  return tf.tidy(() => {
    const predsTensor = model.predict(xs) as tf.Tensor2D;
    const preds = predsTensor.dataSync();
    const targets = ys.dataSync();
    const n = preds.length;
    if (n === 0) return { loss: 0, acc: 0 };

    let totalLoss = 0;
    let correct = 0;

    for (let i = 0; i < n; i++) {
      const y = targets[i];
      const pRaw = preds[i];
      const p = pRaw < 1e-7 ? 1e-7 : pRaw > 1 - 1e-7 ? 1 - 1e-7 : pRaw;

      totalLoss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      if ((pRaw >= 0.5 ? 1 : 0) === y) {
        correct++;
      }
    }

    return {
      loss: totalLoss / n,
      acc: correct / n,
    };
  });
}

/**
 * Precomputes the normalized 2D grid tensor once for a dataset so we don't
 * reallocate JS arrays on every decision boundary update.
 */
export function buildNormalizedGridTensor(
  bounds: { minX1: number; maxX1: number; minX2: number; maxX2: number },
  stats: NormalizationStats,
  resolution: number = 34
): tf.Tensor2D {
  const total = resolution * resolution;
  const flatCoords = new Float32Array(total * 2);
  const stepX1 = (bounds.maxX1 - bounds.minX1) / (resolution - 1);
  const stepX2 = (bounds.maxX2 - bounds.minX2) / (resolution - 1);

  let ptr = 0;
  for (let r = 0; r < resolution; r++) {
    const x2 = bounds.maxX2 - r * stepX2;
    for (let c = 0; c < resolution; c++) {
      const x1 = bounds.minX1 + c * stepX1;
      const [nx1, nx2] = normalizePoint(x1, x2, stats);
      flatCoords[ptr++] = nx1;
      flatCoords[ptr++] = nx2;
    }
  }

  return tf.tensor2d(flatCoords, [total, 2]);
}

/**
 * Computes a 2D probability heatmap over the dataset bounds.
 * Accepts an optional pre-allocated gridTensor for maximum speed during training.
 */
export function predictDecisionGrid(
  model: tf.Sequential | null,
  bounds: { minX1: number; maxX1: number; minX2: number; maxX2: number },
  stats: NormalizationStats,
  resolution: number = 34,
  cachedGridTensor?: tf.Tensor2D | null
): Float32Array {
  if (!model) {
    return new Float32Array(resolution * resolution).fill(0.5);
  }

  return tf.tidy(() => {
    const gridTensor =
      cachedGridTensor && !cachedGridTensor.isDisposed
        ? cachedGridTensor
        : buildNormalizedGridTensor(bounds, stats, resolution);
    const preds = model.predict(gridTensor) as tf.Tensor2D;
    return Float32Array.from(preds.dataSync());
  });
}
