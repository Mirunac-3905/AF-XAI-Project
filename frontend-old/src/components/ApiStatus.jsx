import React, { useEffect, useState } from 'react';
import { checkHealth } from '../services/api';

const ApiStatus = ({ onStatusChange }) => {
  const [status, setStatus] = useState('checking');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const health = await checkHealth();
        setStatus('connected');
        setMessage(`Backend: ${health.service} - ${health.model}`);
        onStatusChange(true);
      } catch (error) {
        setStatus('offline');
        setMessage('Backend is offline or unreachable');
        onStatusChange(false);
      }
    };

    checkBackendStatus();
  }, [onStatusChange]);

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#22c55e';
      case 'offline':
        return '#ef4444';
      default:
        return '#f59e0b';
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.statusRow}>
        <span
          style={{
            ...styles.statusDot,
            backgroundColor: getStatusColor(),
          }}
        />
        <span style={styles.statusText}>
          {status === 'checking' ? 'Checking backend...' : message}
        </span>
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '16px',
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #e5e7eb',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  statusDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  statusText: {
    fontSize: '14px',
    color: '#374151',
  },
};

export default ApiStatus;
