import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { EpochMetric } from '@/types/ml';
import { FrenchTooltip } from '@/components/ui/FrenchTooltip';

interface LearningCurvesChartProps {
  history: EpochMetric[];
  presentationMode?: boolean;
}

const LearningCurvesChartInner: React.FC<LearningCurvesChartProps> = ({
  history,
  presentationMode = false,
}) => {
  const chartHeight = presentationMode ? 240 : 195;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Loss Panel */}
      <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <h4 className="text-sm font-bold text-slate-100">
              Courbe de Coût (Binary Cross-Entropy Loss)
            </h4>
            <FrenchTooltip
              term="Training & Validation Loss"
              explanation="Mesure l’erreur probabiliste du réseau. Si la Training Loss baisse mais que la Validation Loss remonte, votre réseau commence à sur-apprendre (Overfitting)."
              sklearnNote="Équivalent de log_loss(y_true, y_pred_proba)."
            />
          </div>
          {history.length > 0 && (
            <span className="text-xs font-mono text-slate-400">
              Val Loss:{' '}
              <strong className="text-violet-400">
                {history[history.length - 1].valLoss.toFixed(4)}
              </strong>
            </span>
          )}
        </div>

        <div style={{ height: chartHeight }} className="w-full">
          {history.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
              Lancez l’entraînement pour observer l’évolution de la Loss en direct
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <LineChart data={history} margin={{ top: 5, right: 12, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1B2F59" />
                <XAxis
                  dataKey="epoch"
                  stroke="#94A3B8"
                  fontSize={11}
                  tickFormatter={(v) => `Ep ${v}`}
                />
                <YAxis stroke="#94A3B8" fontSize={11} domain={['auto', 'auto']} />
                <Tooltip
                  isAnimationActive={false}
                  contentStyle={{
                    backgroundColor: '#070E1D',
                    borderColor: '#59D9C5',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => value.toFixed(4)}
                  labelFormatter={(label) => `Epoch ${label}`}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  name="Training Loss"
                  stroke="#59D9C5"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  name="Validation Loss"
                  stroke="#FB923C"
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Accuracy Panel */}
      <div className="p-4 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <h4 className="text-sm font-bold text-slate-100">
              Précision de Classification (Accuracy %)
            </h4>
            <FrenchTooltip
              term="Training & Validation Accuracy"
              explanation="Proportion de points bien classés (seuil P(y=1) ≥ 0.5). Comparez toujours la Validation Accuracy à la Training Accuracy pour évaluer la généralisation."
              sklearnNote="Équivalent de accuracy_score(y_true, y_pred)."
            />
          </div>
          {history.length > 0 && (
            <span className="text-xs font-mono text-slate-400">
              Val Acc:{' '}
              <strong className="text-turquoise">
                {(history[history.length - 1].valAcc * 100).toFixed(1)}%
              </strong>
            </span>
          )}
        </div>

        <div style={{ height: chartHeight }} className="w-full">
          {history.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
              Les courbes de Training & Validation Accuracy s’afficheront ici
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <LineChart data={history} margin={{ top: 5, right: 12, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1B2F59" />
                <XAxis
                  dataKey="epoch"
                  stroke="#94A3B8"
                  fontSize={11}
                  tickFormatter={(v) => `Ep ${v}`}
                />
                <YAxis
                  stroke="#94A3B8"
                  fontSize={11}
                  domain={[0, 1]}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                />
                <Tooltip
                  isAnimationActive={false}
                  contentStyle={{
                    backgroundColor: '#070E1D',
                    borderColor: '#986DC7',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                  labelFormatter={(label) => `Epoch ${label}`}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  type="monotone"
                  dataKey="acc"
                  name="Training Accuracy"
                  stroke="#38BDF8"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="valAcc"
                  name="Validation Accuracy"
                  stroke="#986DC7"
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
};

export const LearningCurvesChart = React.memo(LearningCurvesChartInner);
