import React from 'react';

const PredictionCard = ({ prediction }) => {
  if (!prediction) return null;
  const isFault = prediction.predicted_fault === 1;
  return (
    <div className={`result-card ${isFault ? 'result-fault' : 'result-normal'}`}>
      <div className="result-kicker">Prediction Result</div>
      <div className="result-main">
        <div className="result-symbol">{isFault ? '!' : '✓'}</div>
        <div>
          <h2 className="result-title">{isFault ? 'FAULT DETECTED' : 'NORMAL GRID CONDITION'}</h2>
          <p className="result-description">
            {isFault
              ? 'The adaptive aggregated OCSVM classified this measurement as an anomalous grid condition.'
              : 'The adaptive aggregated OCSVM classified this measurement as a normal grid condition.'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
