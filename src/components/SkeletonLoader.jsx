import React from 'react';

export default function SkeletonLoader({ rows = 4, height = '20px' }) {
  const skeletons = Array(rows).fill(0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', width: '100%' }}>
      <style>{`
        @keyframes shimmer {
          0% { background-position: -1000px 0; }
          100% { background-position: 1000px 0; }
        }
        .skeleton-row {
          background: linear-gradient(90deg, var(--surface-color) 25%, var(--border-color) 50%, var(--surface-color) 75%);
          background-size: 1000px 100%;
          animation: shimmer 2s infinite linear;
          border-radius: 4px;
        }
      `}</style>
      {skeletons.map((_, i) => (
        <div 
          key={i} 
          className="skeleton-row" 
          style={{ 
            height, 
            width: i === rows - 1 ? '70%' : '100%' // Make the last row slightly shorter for a staggered look
          }} 
        />
      ))}
    </div>
  );
}
