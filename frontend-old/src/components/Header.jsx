import React from 'react';

const Header = ({ backendConnected }) => {
  return (
    <header style={styles.header}>
      <div style={styles.titleContainer}>
        <h1 style={styles.title}>AF-XAI</h1>
        <p style={styles.subtitle}>Adaptive Federated Explainable AI</p>
        <p style={styles.subtitle}>Smart Grid Fault Detection</p>
      </div>
      <div style={styles.statusContainer}>
        <span
          style={{
            ...styles.statusDot,
            backgroundColor: backendConnected ? '#22c55e' : '#ef4444',
          }}
        />
        <span style={styles.statusText}>
          {backendConnected ? 'Backend Connected' : 'Backend Offline'}
        </span>
      </div>
    </header>
  );
};

const styles = {
  header: {
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '20px',
    marginBottom: '30px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    margin: 0,
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#1f2937',
  },
  subtitle: {
    margin: '5px 0 0 0',
    fontSize: '16px',
    color: '#6b7280',
  },
  statusContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  statusDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
  },
  statusText: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151',
  },
};

export default Header;
