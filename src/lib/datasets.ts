import { DataPoint, DatasetType, GeneratedDataset, NormalizationStats } from '@/types/ml';

/**
 * Deterministic PRNG (Mulberry32) so datasets are 100% reproducible given a seed.
 */
export class SeededRandom {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed >>> 0;
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Standard Normal variate using Box-Muller transform
   */
  nextGaussian(mean = 0, std = 1): number {
    const u1 = Math.max(1e-10, this.next());
    const u2 = this.next();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * std + mean;
  }

  shuffle<T>(array: T[]): T[] {
    const copy = [...array];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }
}

export function normalizePoint(
  x1: number,
  x2: number,
  stats: NormalizationStats
): [number, number] {
  return [
    (x1 - stats.meanX1) / (stats.stdX1 || 1),
    (x2 - stats.meanX2) / (stats.stdX2 || 1),
  ];
}

export function generateDataset(
  type: DatasetType,
  numSamples: number = 400,
  noise: number = 0.15,
  trainRatio: number = 0.8,
  seed: number = 1337
): GeneratedDataset {
  const rng = new SeededRandom(seed);
  const class0: Omit<DataPoint, 'split'>[] = [];
  const class1: Omit<DataPoint, 'split'>[] = [];

  const n0 = Math.floor(numSamples / 2);
  const n1 = numSamples - n0;

  if (type === 'moons') {
    for (let i = 0; i < n0; i++) {
      const angle = (Math.PI * i) / Math.max(1, n0 - 1);
      const x1 = Math.cos(angle) + rng.nextGaussian(0, noise);
      const x2 = Math.sin(angle) + rng.nextGaussian(0, noise);
      class0.push({ x1, x2, label: 0 });
    }
    for (let i = 0; i < n1; i++) {
      const angle = (Math.PI * i) / Math.max(1, n1 - 1);
      const x1 = 1 - Math.cos(angle) + rng.nextGaussian(0, noise);
      const x2 = 1 - Math.sin(angle) - 0.5 + rng.nextGaussian(0, noise);
      class1.push({ x1, x2, label: 1 });
    }
  } else if (type === 'circles') {
    const innerFactor = 0.45;
    for (let i = 0; i < n0; i++) {
      const angle = (2 * Math.PI * i) / n0;
      const x1 = Math.cos(angle) + rng.nextGaussian(0, noise);
      const x2 = Math.sin(angle) + rng.nextGaussian(0, noise);
      class0.push({ x1, x2, label: 0 });
    }
    for (let i = 0; i < n1; i++) {
      const angle = (2 * Math.PI * i) / n1;
      const x1 = innerFactor * Math.cos(angle) + rng.nextGaussian(0, noise);
      const x2 = innerFactor * Math.sin(angle) + rng.nextGaussian(0, noise);
      class1.push({ x1, x2, label: 1 });
    }
  } else if (type === 'xor') {
    for (let i = 0; i < n0; i++) {
      const quadrant = i % 2 === 0 ? 1 : -1;
      const x1 = quadrant * (0.65 + rng.next() * 0.6) + rng.nextGaussian(0, noise);
      const x2 = quadrant * (0.65 + rng.next() * 0.6) + rng.nextGaussian(0, noise);
      class0.push({ x1, x2, label: 0 });
    }
    for (let i = 0; i < n1; i++) {
      const quadrant = i % 2 === 0 ? 1 : -1;
      const x1 = quadrant * (0.65 + rng.next() * 0.6) + rng.nextGaussian(0, noise);
      const x2 = -quadrant * (0.65 + rng.next() * 0.6) + rng.nextGaussian(0, noise);
      class1.push({ x1, x2, label: 1 });
    }
  } else {
    // Linearly separable
    for (let i = 0; i < n0; i++) {
      const x1 = -0.8 + rng.nextGaussian(0, 0.45 + noise);
      const x2 = 0.7 + rng.nextGaussian(0, 0.45 + noise);
      class0.push({ x1, x2, label: 0 });
    }
    for (let i = 0; i < n1; i++) {
      const x1 = 0.8 + rng.nextGaussian(0, 0.45 + noise);
      const x2 = -0.7 + rng.nextGaussian(0, 0.45 + noise);
      class1.push({ x1, x2, label: 1 });
    }
  }

  // Stratified train/validation split
  const shuffled0 = rng.shuffle(class0);
  const shuffled1 = rng.shuffle(class1);

  const trainCount0 = Math.max(1, Math.round(n0 * trainRatio));
  const trainCount1 = Math.max(1, Math.round(n1 * trainRatio));

  const trainRaw: DataPoint[] = [
    ...shuffled0.slice(0, trainCount0).map((p) => ({ ...p, split: 'train' as const })),
    ...shuffled1.slice(0, trainCount1).map((p) => ({ ...p, split: 'train' as const })),
  ];

  const valRaw: DataPoint[] = [
    ...shuffled0.slice(trainCount0).map((p) => ({ ...p, split: 'val' as const })),
    ...shuffled1.slice(trainCount1).map((p) => ({ ...p, split: 'val' as const })),
  ];

  const trainPoints = rng.shuffle(trainRaw);
  const valPoints = rng.shuffle(valRaw);
  const points = [...trainPoints, ...valPoints];

  // Compute normalization statistics strictly on TRAINING data (prevent data leakage!)
  const meanX1 =
    trainPoints.reduce((acc, p) => acc + p.x1, 0) / Math.max(1, trainPoints.length);
  const meanX2 =
    trainPoints.reduce((acc, p) => acc + p.x2, 0) / Math.max(1, trainPoints.length);

  const varX1 =
    trainPoints.reduce((acc, p) => acc + (p.x1 - meanX1) ** 2, 0) /
    Math.max(1, trainPoints.length);
  const varX2 =
    trainPoints.reduce((acc, p) => acc + (p.x2 - meanX2) ** 2, 0) /
    Math.max(1, trainPoints.length);

  const stats: NormalizationStats = {
    meanX1,
    stdX1: Math.sqrt(varX1) || 1,
    meanX2,
    stdX2: Math.sqrt(varX2) || 1,
  };

  let minX1 = Infinity;
  let maxX1 = -Infinity;
  let minX2 = Infinity;
  let maxX2 = -Infinity;

  for (const p of points) {
    if (p.x1 < minX1) minX1 = p.x1;
    if (p.x1 > maxX1) maxX1 = p.x1;
    if (p.x2 < minX2) minX2 = p.x2;
    if (p.x2 > maxX2) maxX2 = p.x2;
  }

  const padX1 = (maxX1 - minX1) * 0.18 || 0.5;
  const padX2 = (maxX2 - minX2) * 0.18 || 0.5;

  return {
    points,
    trainPoints,
    valPoints,
    stats,
    bounds: {
      minX1: minX1 - padX1,
      maxX1: maxX1 + padX1,
      minX2: minX2 - padX2,
      maxX2: maxX2 + padX2,
    },
  };
}
