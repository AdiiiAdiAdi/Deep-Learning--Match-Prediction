import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
import { GLOBAL_FEATURES, TEAMS } from '../mockData';
import { getFeatureImportance } from '../api/client';
import { useApi } from '../hooks/useApi';
import SkeletonLoader from './SkeletonLoader';
import ErrorCard from './ErrorCard';

export default function FeatureImportance() {
  const { data, loading, error, refetch } = useApi(getFeatureImportance, null, []);

  const CustomGlobalTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const pData = payload[0].payload;
      return (
        <div style={{ backgroundColor: 'var(--surface-color)', border: '1px solid var(--border-color)', padding: '10px', borderRadius: '4px' }}>
          <p className="text-sm font-medium mb-1">{pData.feature_name || pData.name}</p>
          <p className="mono text-xs" style={{ color: pData.importance < 0 ? '#888' : (pData.type === 'sequence' ? 'var(--color-info)' : 'var(--color-win)') }}>
            Score: {typeof pData.importance === 'number' ? pData.importance.toFixed(4) : pData.score}
            {pData.importance < 0 && ' (Negative importance means this feature adds noise)'}
          </p>
        </div>
      );
    }
    return null;
  };

  if (loading) return <SkeletonLoader rows={8} height="60px" />;

  let chartData = data || GLOBAL_FEATURES.map(v => ({ feature_name: v.name, importance: v.score, type: v.group === 'Goals' || v.group === 'Discipline' ? 'sequence' : 'static' }));
  
  // Sort by importance descending
  chartData = [...chartData].sort((a, b) => b.importance - a.importance);

  return (
    <div className="flex-col gap-4 h-full" style={{ height: 'calc(100vh - 120px)' }}>
       {error && <ErrorCard message={error.message} onRetry={refetch} />}
      
      <div className="card h-full" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
           <div>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>Permutation Feature Importance</h2>
              <p className="text-sm text-secondary">Average accuracy drop when feature is shuffled across test set.</p>
           </div>
           
           <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', maxWidth: '400px', justifyContent: 'flex-end' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <div style={{ width: '10px', height: '10px', backgroundColor: 'var(--color-info)', borderRadius: '2px' }} />
                <span className="text-xs text-secondary">Sequence Matrix (LSTM)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <div style={{ width: '10px', height: '10px', backgroundColor: 'var(--color-win)', borderRadius: '2px' }} />
                <span className="text-xs text-secondary">Static Tensors (ANN)</span>
              </div>
           </div>
        </div>
        
        <div style={{ flexGrow: 1, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 160, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" horizontal={true} vertical={false} />
              <XAxis type="number" stroke="var(--text-secondary)" tick={{fontSize: 12}} />
              <YAxis dataKey={data ? "feature_name" : "name"} type="category" stroke="var(--text-secondary)" tick={{fontSize: 11}} width={150} />
              <Tooltip cursor={{ fill: 'var(--border-color)', opacity: 0.2 }} content={<CustomGlobalTooltip />} />
              <Bar dataKey={data ? "importance" : "score"} radius={[0, 4, 4, 0]} barSize={20}>
                {chartData.map((entry, index) => {
                  let color = entry.type === 'sequence' ? 'var(--color-info)' : 'var(--color-win)';
                  if (entry.importance < 0) color = '#444'; // negative importance styling
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
