import React from 'react';

const ScoreCard = ({ prediction }) => {
  if (!prediction) return null;

  return (
    <div style={styles.container}>
      <h2 style={styles.sectionTitle}>Scores</h2>
      <div style={styles.scoreGrid}>
        <div style={styles.scoreItem}>
          <div style={styles.scoreLabel}>Decision Score</div>
          <div style={styles.scoreValue}>
            {prediction.decision_score.toFixed(4)}
          </div>
        </div>
        <div style={styles.scoreItem}>
          <div style={styles.scoreLabel}>Anomaly Score</div>
          <div style={styles.scoreValue}>
            {prediction.anomaly_score.toFixed(4)}
          </div>
        </div>
      </div>
      <div style={styles.note}>
        <em>
          Anomaly score is derived from the model decision function; it is not a
          probability.
        </em>
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
  scoreGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '20px',
    marginBottom: '15px',
  },
  scoreItem: {
    padding: '20px',
    backgroundColor: '#f9fafb',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
  },
  scoreLabel: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#6b7280',
    marginBottom: '8px',
  },
  scoreValue: {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#1f2937',
  },
  note: {
    fontSize: '12px',
    color: '#6b7280',
    marginTop: '10px',
  },
};

export default ScoreCard;
