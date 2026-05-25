import React from 'react';
import { AlertOctagon, RefreshCw } from 'lucide-react';

export default function ErrorCard({ message, onRetry }) {
  return (
    <div style={{
      border: '1px solid var(--color-loss)',
      backgroundColor: 'rgba(248, 81, 73, 0.1)',
      padding: '1.5rem',
      borderRadius: '4px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '1rem',
      width: '100%',
      margin: '1rem 0'
    }}>
      <AlertOctagon size={32} color="var(--color-loss)" />
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ color: 'var(--color-loss)', fontSize: '1rem', marginBottom: '0.5rem' }}>Failed to load data</h3>
        <p className="text-secondary text-sm">{message}</p>
      </div>
      {onRetry && (
        <button 
          onClick={onRetry}
          className="btn-primary" 
          style={{ 
            width: 'auto', 
            padding: '0.5rem 1.5rem', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem',
            borderColor: 'var(--color-loss)'
          }}
        >
          <RefreshCw size={14} /> Retry
        </button>
      )}
    </div>
  );
}
