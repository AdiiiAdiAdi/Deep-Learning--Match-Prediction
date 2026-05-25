import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { CheckCircle2, TrendingUp, AlertCircle, ArrowUpCircle } from 'lucide-react';
import { MODEL_METRICS, CONFUSION_MATRIX, EPOCH_DATA } from '../mockData';
import { getMetrics, getTrainingHistory } from '../api/client';
import SkeletonLoader from './SkeletonLoader';
import ErrorCard from './ErrorCard';

export default function ModelPerformance() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [epochData, setEpochData] = useState(null);

  const fetchAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, h] = await Promise.all([getMetrics(), getTrainingHistory()]);
      setMetricsData(m);
      setEpochData(h);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAll();
  }, []);

  if (loading) return <SkeletonLoader rows={5} height="150px" />;

  const mData = metricsData || {
    accuracy: parseFloat(MODEL_METRICS.accuracy.value) / 100,
    macro: { precision: 0.652, recall: 0.648, f1: 0.650 },
    confusion_matrix: CONFUSION_MATRIX
  };
  const hData = epochData || EPOCH_DATA;
  
  // Find minimum val_loss
  let bestEpoch = 34;
  if (epochData && epochData.length > 0) {
    let minLoss = Infinity;
    epochData.forEach(e => {
       if (e.val_loss < minLoss) { minLoss = e.val_loss; bestEpoch = e.epoch; }
    });
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ backgroundColor: 'var(--surface-color)', border: '1px solid var(--border-color)', padding: '10px', borderRadius: '4px' }}>
          <p className="mono text-sm mb-2">Epoch {label}</p>
          {payload.map((p, i) => (
             <p key={i} className="mono text-xs" style={{ color: p.color }}>
               {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
             </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const getConfusionCellColor = (rowIndex, colIndex, value, maxVal) => {
    const intensity = Math.max(0.1, value / (maxVal || 1));
    if (rowIndex === colIndex) return `rgba(57, 211, 83, ${Math.max(0.3, intensity)})`;
    return `rgba(248, 81, 73, ${intensity})`;
  };

  const cnLabels = ['Away Win', 'Draw', 'Home Win']; // Correct matching output

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(400px, 1fr) 1.5fr', gap: '2rem', height: '100%' }}>
      
      {/* Left Column */}
      <div className="flex-col gap-4">
        {error && <ErrorCard message={error.message} onRetry={fetchAll} />}
        
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Test Set Evaluation</h2>
        
        {/* Metric Cards Row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
          {[
            { label: 'Accuracy', value: `${(mData.accuracy * 100).toFixed(1)}%`, icon: <CheckCircle2 color="var(--color-win)" /> },
            { label: 'Precision (macro)', value: `${(mData.macro.precision * 100).toFixed(1)}%`, icon: <TrendingUp color="var(--color-info)" /> },
            { label: 'Recall (macro)', value: `${(mData.macro.recall * 100).toFixed(1)}%`, icon: <AlertCircle color="var(--color-draw)" /> },
            { label: 'F1 Score', value: `${(mData.macro.f1 * 100).toFixed(1)}%`, icon: <ArrowUpCircle color="var(--color-embeddings)" /> }
          ].map((metric, i) => (
            <div key={i} className="card">
              <div className="text-secondary text-sm flex justify-between items-center mb-2">
                 <span>{metric.label}</span>
                 {metric.icon}
              </div>
              <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600 }}>{metric.value}</div>
            </div>
          ))}
        </div>

        {/* Confusion Matrix */}
        <div className="card">
          <h3 className="text-sm text-secondary mb-4">Confusion Matrix (Holdout Set)</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '100px repeat(3, 1fr)', gap: '4px' }}>
            <div />
            {cnLabels.map(l => (
              <div key={`col-${l}`} className="text-xs text-secondary mono" style={{ textAlign: 'center', padding: '0.5rem' }}>Pred {l}</div>
            ))}
            
            {mData.confusion_matrix.map((row, rIdx) => {
              const rowMax = Math.max(...row);
              return (
                <React.Fragment key={`row-${rIdx}`}>
                  <div className="text-xs text-secondary mono" style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '0.5rem' }}>
                    True {cnLabels[rIdx]}
                  </div>
                  {row.map((val, cIdx) => (
                    <div 
                      key={`cell-${rIdx}-${cIdx}`}
                      style={{ 
                        background: getConfusionCellColor(rIdx, cIdx, val, rowMax),
                        color: rIdx === cIdx && (val / (rowMax || 1)) > 0.5 ? '#000' : '#fff',
                        padding: '1rem',
                        borderRadius: '4px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: rIdx === cIdx ? '1px solid var(--color-win)' : '1px solid transparent'
                      }}
                    >
                      <span className="mono font-semibold text-lg">{val}</span>
                    </div>
                  ))}
                </React.Fragment>
              );
            })}
          </div>
          
        </div>
      </div>

      {/* Right Column: Training History */}
      <div className="card flex-col gap-4">
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Loss Curvature & Early Stopping</h2>
        
        {/* Loss Chart */}
        <div style={{ height: '280px', width: '100%', marginBottom: '2rem' }}>
           <h3 className="text-sm text-secondary mb-4">Training vs Validation Loss</h3>
           <ResponsiveContainer width="100%" height="100%">
             <LineChart data={hData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
               <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
               <XAxis dataKey="epoch" stroke="var(--text-secondary)" tick={{fontSize: 12}} />
               <YAxis stroke="var(--text-secondary)" tick={{fontSize: 12}} domain={['auto', 'auto']} />
               <Tooltip content={<CustomTooltip />} />
               <Legend iconType="circle" />
               <ReferenceLine x={bestEpoch} stroke="var(--text-secondary)" strokeDasharray="3 3" label={{ position: 'top', value: `Best (Epoch ${bestEpoch})`, fill: 'var(--text-secondary)', fontSize: 12 }} />
               <Line type="monotone" dataKey={metricsData ? 'loss' : 'trainLoss'} name="Train Loss" stroke="var(--color-info)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
               <Line type="monotone" dataKey={metricsData ? 'val_loss' : 'valLoss'} name="Val Loss" stroke="var(--color-draw)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
             </LineChart>
           </ResponsiveContainer>
        </div>

        {/* Acc Chart */}
        <div style={{ height: '280px', width: '100%' }}>
           <h3 className="text-sm text-secondary mb-4">Training vs Validation Accuracy (%)</h3>
           <ResponsiveContainer width="100%" height="100%">
             <LineChart data={hData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
               <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
               <XAxis dataKey="epoch" stroke="var(--text-secondary)" tick={{fontSize: 12}} />
               <YAxis stroke="var(--text-secondary)" tick={{fontSize: 12}} domain={[0, 1]} />
               <Tooltip content={<CustomTooltip />} />
               <Legend iconType="circle" />
               <ReferenceLine x={bestEpoch} stroke="var(--text-secondary)" strokeDasharray="3 3" />
               <Line type="monotone" dataKey={metricsData ? 'accuracy' : 'trainAcc'} name="Train Accuracy" stroke="var(--color-win)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
               <Line type="monotone" dataKey={metricsData ? 'val_accuracy' : 'valAcc'} name="Val Accuracy" stroke="var(--color-loss)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
             </LineChart>
           </ResponsiveContainer>
        </div>
        
      </div>

    </div>
  );
}
