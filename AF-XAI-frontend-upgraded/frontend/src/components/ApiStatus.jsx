import React, { useEffect, useState } from 'react';
import { checkHealth } from '../services/api';

const ApiStatus = ({ onStatusChange }) => {
  const [status, setStatus] = useState('checking');
  const [message, setMessage] = useState('Checking prediction service...');

  useEffect(() => {
    let active = true;
    const checkBackendStatus = async () => {
      try {
        const health = await checkHealth();
        if (!active) return;
        setStatus('connected');
        setMessage(`${health.service} · ${health.model}`);
        onStatusChange(true);
      } catch {
        if (!active) return;
        setStatus('offline');
        setMessage('Prediction API is offline or unreachable');
        onStatusChange(false);
      }
    };
    checkBackendStatus();
    return () => { active = false; };
  }, [onStatusChange]);

  const color = status === 'connected' ? '#22a866' : status === 'offline' ? '#d74a43' : '#c88b22';

  return (
    <div className="api-strip">
      <div className="api-left">
        <span className="status-dot" style={{ background: color }} />
        <span className="api-text">{message}</span>
      </div>
      <span className="api-badge" style={status === 'offline' ? { background:'#fff0ef', color:'#b63830' } : undefined}>
        {status === 'connected' ? 'MODEL READY' : status === 'offline' ? 'OFFLINE' : 'CHECKING'}
      </span>
    </div>
  );
};

export default ApiStatus;
