import React from 'react';

const PredictionCard = ({ prediction }) => {
  if (!prediction) return null;

  const isFault = prediction.predicted_fault === 1;

  return (
    <div style={styles.container}>
      <h2 style={styles.sectionTitle}>Prediction</h2>
      <div
        style={{
          ...styles.resultCard,
          backgroundColor: isFault ? '#fef2f2' : '#f0fdf4',
          borderColor: isFault ? '#ef4444' : '#22c55e',
        }}
      >
        <div
          style={{
            ...styles.resultText,
            color: isFault ? '#dc2626' : '#16a34a',
          }}
        >
          {isFault ? 'FAULT DETECTED' : 'NORMAL CONDITION'}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    marginBottom: '30px',
  },
  sectionTitle: {
    fontSize: '20px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '20px',
    borderBottom: '2px solid #e5e7eb',
    paddingBottom: '10px',
  },
  resultCard: {
    padding: '30px',
    borderRadius: '12px',
    border: '2px solid',
    textAlign: 'center',
  },
  resultText: {
    fontSize: '28px',
    fontWeight: 'bold',
    margin: 0,
  },
};

export default PredictionCard;
