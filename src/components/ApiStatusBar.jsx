import React, { useState, useEffect } from 'react';
import { healthCheck } from '../api/client';

export default function ApiStatusBar() {
  const [status, setStatus] = useState('loading'); // 'loading' | 'healthy' | 'no-model' | 'offline'

  const checkHealth = async () => {
    try {
      const data = await healthCheck();
      if (data.status === 'ok' && data.model_loaded) {
        setStatus('healthy');
      } else {
        setStatus('no-model');
      }
    } catch (err) {
      setStatus('offline');
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // recheck every 30s
    return () => clearInterval(interval);
  }, []);

  const getStatusStyles = () => {
    switch (status) {
      case 'healthy':
        return { dot: '#39d353', text: 'API connected — model loaded' };
      case 'no-model':
        return { dot: '#d29922', text: 'API running but model not loaded' };
      case 'offline':
      default:
        return { dot: '#f85149', text: 'Flask API offline — run: python src/app.py' };
    }
  };

  const { dot, text } = getStatusStyles();

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '28px',
      backgroundColor: 'var(--surface-color)',
      borderBottom: '1px solid var(--border-color)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      zIndex: 9999,
      fontSize: '0.75rem',
      fontFamily: 'var(--font-mono)',
      color: 'var(--text-secondary)'
    }}>
      {status === 'loading' ? (
        <span>Checking API connection...</span>
      ) : (
        <>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: dot }} />
          <span>{text}</span>
        </>
      )}
    </div>
  );
}
