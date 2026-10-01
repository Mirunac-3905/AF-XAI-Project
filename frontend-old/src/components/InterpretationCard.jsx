import React from 'react';

const InterpretationCard = ({ explanations }) => {
  if (!explanations || !explanations.length) return null;

  const featureLabels = {
    voltage: 'Voltage',
    frequency: 'Frequency',
    active_power: 'Active Power',
    reactive_power: 'Reactive Power',
    load_demand: 'Load Demand',
    renewable_output: 'Renewable Output',
    temperature: 'Temperature',
    humidity: 'Humidity',
    power_loss: 'Power Loss',
    stability_index: 'Stability Index',
  };

  const sortedExplanations = [...explanations].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
  );

  const topFeatures = sortedExplanations.slice(0, 3);

  return (
    <div style={styles.container}>
      <h2 style={styles.sectionTitle}>Interpretation</h2>
      <div style={styles.content}>
        <p style={styles.mainText}>
          The features with the largest absolute SHAP contributions had the
          strongest influence on the model's anomaly score for this prediction.
        </p>
        <div style={styles.topFeaturesList}>
          <h4 style={styles.subTitle}>Top Contributing Features:</h4>
          {topFeatures.map((item, index) => (
            <div key={item.feature} style={styles.featureItem}>
              <span style={styles.featureNumber}>{index + 1}.</span>
              <span style={styles.featureName}>
                {featureLabels[item.feature] || item.feature}
              </span>
              <span style={styles.featureValue}>
                (SHAP: {item.shap_value.toFixed(4)})
              </span>
            </div>
          ))}
        </div>
        <p style={styles.note}>
          <em>
            SHAP values represent the contribution to the model anomaly score,
            not the cause of the fault.
          </em>
        </p>
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
  content: {
    backgroundColor: '#f9fafb',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
  },
  mainText: {
    fontSize: '15px',
    color: '#374151',
    lineHeight: '1.6',
    marginBottom: '20px',
  },
  subTitle: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: '12px',
    marginTop: 0,
  },
  topFeaturesList: {
    marginBottom: '20px',
  },
  featureItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
    fontSize: '14px',
    color: '#374151',
  },
  featureNumber: {
    fontWeight: '600',
    color: '#6b7280',
  },
  featureName: {
    fontWeight: '500',
    color: '#1f2937',
  },
  featureValue: {
    color: '#6b7280',
    fontSize: '13px',
  },
  note: {
    fontSize: '12px',
    color: '#6b7280',
    margin: 0,
  },
};

export default InterpretationCard;
