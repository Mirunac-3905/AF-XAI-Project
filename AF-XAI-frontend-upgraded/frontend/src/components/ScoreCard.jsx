import React from 'react';

const ScoreCard = ({ prediction }) => {
  if (!prediction) return null;
  return (
    <div className="score-grid">
      <div className="score-card">
        <div className="score-label">Decision Score</div>
        <div className="score-value">{prediction.decision_score.toFixed(4)}</div>
        <div className="score-note">Negative values cross the OCSVM fault threshold.</div>
      </div>
      <div className="score-card">
        <div className="score-label">Anomaly Score</div>
        <div className="score-value">{prediction.anomaly_score.toFixed(4)}</div>
        <div className="score-note">Derived as −decision score; this is not a probability.</div>
      </div>
    </div>
  );
};

export default ScoreCard;
