import React from 'react';

const labels = {
  voltage: 'Voltage', frequency: 'Frequency', active_power: 'Active Power',
  reactive_power: 'Reactive Power', load_demand: 'Load Demand', renewable_output: 'Renewable Output',
  temperature: 'Temperature', humidity: 'Humidity', power_loss: 'Power Loss', stability_index: 'Stability Index',
};

const InterpretationCard = ({ explanations }) => {
  if (!explanations?.length) return null;
  const top = [...explanations].sort((a,b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 5);
  return (
    <div className="interpretation-grid">
      <div className="interpretation-card">
        <h3>Top Contributing Features</h3>
        {top.map((item) => (
          <div className="top-feature" key={item.feature}>
            <span className="feature-name">{labels[item.feature] || item.feature}</span>
            <span className={`feature-shap ${item.shap_value >= 0 ? 'positive' : 'negative'}`}>
              {item.shap_value >= 0 ? '+' : ''}{item.shap_value.toFixed(4)}
            </span>
          </div>
        ))}
      </div>
      <div className="interpretation-card">
        <h3>How to Read This Explanation</h3>
        <p>
          SHAP values describe how each feature contributed to the model&apos;s anomaly score for this individual prediction.
          Positive contributions push the score toward anomaly, while negative contributions push it toward normal.
        </p>
        <p style={{ marginTop: 12 }}>
          <strong>Note:</strong> a SHAP contribution indicates model influence, not a causal explanation of the physical fault.
        </p>
      </div>
    </div>
  );
};

export default InterpretationCard;
