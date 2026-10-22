export type DatasetType = 'moons' | 'circles' | 'xor' | 'linear';

export type ActivationType = 'relu' | 'tanh' | 'sigmoid';
export type NeuronActivationType = 'linear' | 'relu' | 'sigmoid' | 'tanh';

export interface DataPoint {
  x1: number;
  x2: number;
  label: 0 | 1;
  split: 'train' | 'val';
}

export interface NormalizationStats {
  meanX1: number;
  stdX1: number;
  meanX2: number;
  stdX2: number;
}

export interface GeneratedDataset {
  points: DataPoint[];
  trainPoints: DataPoint[];
  valPoints: DataPoint[];
  stats: NormalizationStats;
  bounds: {
    minX1: number;
    maxX1: number;
    minX2: number;
    maxX2: number;
  };
}

export interface ModelConfig {
  hiddenLayers: number; // 0 to 4
  neuronsPerLayer: number; // 1 to 16
  activation: ActivationType;
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface EpochMetric {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
}

export interface LayerWeights {
  layerIndex: number;
  layerName: string;
  inputSize: number;
  outputSize: number;
  weights: number[][]; // [inputIdx][outputIdx]
  biases: number[];    // [outputIdx]
}

export interface ModelComparisonSnapshot {
  id: string;
  name: string;
  timestamp: string;
  datasetType: DatasetType;
  noise: number;
  config: ModelConfig;
  trainableParams: number;
  finalEpoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  history: EpochMetric[];
}

export interface GuidedExperiment {
  id: 'A' | 'B' | 'C' | 'D';
  title: string;
  subtitle: string;
  dataset: DatasetType;
  noise: number;
  samples: number;
  trainSplit: number;
  config: ModelConfig;
  explanationFr: string;
  questionFr: string;
  optionsFr: string[];
  correctOptionIndex: number;
  takeawayFr: string;
}
