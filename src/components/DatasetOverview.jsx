import React, { useState } from 'react';
import { Database, Activity, Type, ArrowRight, ShieldCheck, BarChartHorizontal } from 'lucide-react';
import { DATASET_STATS, PIPELINE_STEPS, QUALITY_HEATMAP } from '../mockData';
import { getDatasetStats } from '../api/client';
import { useApi } from '../hooks/useApi';
import SkeletonLoader from './SkeletonLoader';
import ErrorCard from './ErrorCard';

export default function DatasetOverview() {
  const [activeStep, setActiveStep] = useState(null);
  const { data, loading, error, refetch } = useApi(getDatasetStats, null, []);

  const getHeatmapColor = (value, max) => {
    const ratio = Math.min(value / max, 1);
    const r = Math.round(57 + (248 - 57) * ratio);
    const g = Math.round(211 + (81 - 211) * ratio);
    const b = Math.round(83 + (73 - 83) * ratio);
    return `rgba(${r}, ${g}, ${b}, 0.8)`;
  };

  if (loading) return <SkeletonLoader rows={6} height="120px" />;

  const stats = data || {
    total_rows: DATASET_STATS.totalMatches.replace(',', ''),
    total_columns: DATASET_STATS.totalFeatures,
    seasons: "12 (2012-2024)",
    null_count: 847,
    dirty_cell_count: 847,
    feature_categories: { null_percentages: {} },
    class_distribution: { H: 45, D: 25, A: 30 }
  };

  const formattedStats = {
    totalMatches: stats.total_rows.toLocaleString(),
    totalFeatures: stats.total_columns,
    seasons: stats.seasons || "12 seasons",
    nullCellPct: stats.total_rows ? ((stats.null_count / (stats.total_rows * stats.total_columns)) * 100).toFixed(2) + '%' : DATASET_STATS.nullCellPct
  };

  return (
    <div className="flex-col gap-4">
      {error && <ErrorCard message={error.message} onRetry={refetch} />}
      
      {/* Top Stats Strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
        {[
          { label: 'Total Matches', value: formattedStats.totalMatches, icon: <Database size={16} /> },
          { label: 'Total Features', value: formattedStats.totalFeatures, icon: <BarChartHorizontal size={16} /> },
          { label: 'Seasons Covered', value: formattedStats.seasons, icon: <Activity size={16} /> },
          { label: 'Null Cell Ratio', value: formattedStats.nullCellPct, icon: <Type size={16} /> }
        ].map((stat, i) => (
          <div key={i} className="card">
             <div className="text-secondary text-sm flex items-center gap-2 mb-2">{stat.icon}{stat.label}</div>
             <div className="mono" style={{ fontSize: '1.5rem', fontWeight: 600 }}>{stat.value}</div>
          </div>
        ))}
      </div>

      <div className="card w-full mb-4">
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}>Preprocessing Pipeline (t-0 forward pass)</h2>
        
        {/* Stepper */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
          {PIPELINE_STEPS.map((step, idx) => {
            // Inject real dynamic data into specific step descriptions
            let finalDesc = step.desc;
            if (step.id === 1 && data) finalDesc = `Loaded ${stats.seasons || '12'} seasons of EPL match and event data. (${stats.total_rows} rows)`;
            if (step.id === 2 && data) finalDesc = `Fixed ${stats.null_count} null cells, imputed with column median or prev value.`;
            
            return (
              <React.Fragment key={step.id}>
                <div 
                  onClick={() => setActiveStep(activeStep === step.id ? null : step.id)}
                  style={{ 
                    flex: 1, 
                    textAlign: 'center',
                    cursor: 'pointer',
                    padding: '1rem',
                    borderRadius: '4px',
                    border: activeStep === step.id ? '1px solid var(--color-info)' : '1px solid var(--border-color)',
                    background: activeStep === step.id ? 'rgba(88, 166, 255, 0.1)' : 'var(--bg-color)',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div className="mono text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>STEP {step.id}</div>
                  <div style={{ fontWeight: 500, marginBottom: '0.5rem', fontSize: '0.875rem' }}>{step.name}</div>
                  <div style={{ 
                    display: 'inline-block',
                    fontSize: '0.75rem',
                    padding: '0.125rem 0.5rem',
                    borderRadius: '12px',
                    background: 'var(--color-win)',
                    color: '#000',
                    fontWeight: 600
                  }}>
                    {step.status}
                  </div>
                </div>
                {idx < PIPELINE_STEPS.length - 1 && (
                  <ArrowRight size={24} color="var(--text-secondary)" style={{ margin: '0 0.5rem' }} />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Expanded Panel */}
        {activeStep && (
          <div style={{ background: 'var(--bg-color)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '1.5rem', animation: 'fadeIn 0.2s ease-in' }}>
            <h3 className="text-sm text-secondary mb-2">{PIPELINE_STEPS.find(s => s.id === activeStep)?.name} Logs</h3>
            <p className="text-sm mb-4">
               {activeStep === 1 && data ? `Loaded ${stats.seasons || '12'} seasons of EPL match and event data. (${stats.total_rows} rows)` : 
                activeStep === 2 && data ? `Fixed ${stats.null_count} null cells, imputed with column median or prev value.` :
                PIPELINE_STEPS.find(s => s.id === activeStep)?.desc}
            </p>
            
            {/* Mock Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }} className="mono">
                <thead>
                  <tr style={{ background: 'var(--surface-color)', textAlign: 'left' }}>
                    <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>index</th>
                    <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>feature_col</th>
                    <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>before</th>
                    <th style={{ padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>after</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)' }}>2841</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)' }}>home_xG_L5</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)', color: 'var(--color-loss)' }}>NaN</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)', color: 'var(--color-win)' }}>1.14</td>
                  </tr>
                  <tr>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)' }}>2842</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)' }}>away_manager_id</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)', color: 'var(--color-loss)' }}>"M. Arteta"</td>
                     <td style={{ padding: '0.5rem', borderBottom: '1px dashed var(--border-color)', color: 'var(--color-win)' }}>42</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Heatmap Section */}
      <div className="card w-full mb-4">
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
          <ShieldCheck size={20} color="var(--color-info)"/> Data Quality Matrix
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 1fr) repeat(4, 1fr)', gap: '4px' }}>
          {/* Headers */}
          <div className="text-xs text-secondary mb-2" style={{ padding: '0.5rem' }}>Feature Category</div>
          <div className="text-xs text-secondary mb-2" style={{ padding: '0.5rem', textAlign: 'center' }}>Null %</div>
          <div className="text-xs text-secondary mb-2" style={{ padding: '0.5rem', textAlign: 'center' }}>Type Errors</div>
          <div className="text-xs text-secondary mb-2" style={{ padding: '0.5rem', textAlign: 'center' }}>Outliers</div>
          <div className="text-xs text-secondary mb-2" style={{ padding: '0.5rem', textAlign: 'center' }}>Inconsistent Labels</div>
          
          {/* Rows */}
          {QUALITY_HEATMAP.map((row, idx) => {
             // Dynamic interpolation if API has it, otherwise fallback
             const actualNulls = (data && stats.feature_categories?.null_percentages?.[row.category.toLowerCase()]) || row.nulls;
             return (
              <React.Fragment key={idx}>
                 <div className="text-sm font-medium" style={{ display: 'flex', alignItems: 'center', padding: '0.5rem', borderRight: '1px solid var(--border-color)' }}>
                   {row.category}
                 </div>
                 
                 <div className="mono flex items-center justify-center text-sm" style={{ 
                   background: getHeatmapColor(actualNulls, 8), 
                   color: actualNulls > 4 ? '#fff' : '#000',
                   borderRadius: '2px'
                 }}>
                   {typeof actualNulls === 'number' ? actualNulls.toFixed(1) : parseFloat(actualNulls).toFixed(1)}%
                 </div>
                 
                 <div className="mono flex items-center justify-center text-sm" style={{ 
                   background: getHeatmapColor(row.types, 3), 
                   color: row.types > 1.5 ? '#fff' : '#000',
                   borderRadius: '2px'
                 }}>
                   {row.types}%
                 </div>
                 
                 <div className="mono flex items-center justify-center text-sm" style={{ 
                   background: getHeatmapColor(row.outliers, 10), 
                   color: row.outliers > 5 ? '#fff' : '#000',
                   borderRadius: '2px'
                 }}>
                   {row.outliers}%
                 </div>
                 
                 <div className="mono flex items-center justify-center text-sm" style={{ 
                   background: getHeatmapColor(row.labels, 1), 
                   color: row.labels > 0.5 ? '#fff' : '#000',
                   borderRadius: '2px'
                 }}>
                   {row.labels}%
                 </div>
              </React.Fragment>
             );
          })}
        </div>
        <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '1rem' }} className="text-xs mono">
           <span>Clean Data</span>
           <div style={{ width: '100px', height: '8px', background: 'linear-gradient(90deg, rgba(57, 211, 83, 0.8), rgba(248, 81, 73, 0.8))', borderRadius: '4px' }}></div>
           <span>Dirty Data</span>
        </div>
      </div>

    </div>
  );
}
