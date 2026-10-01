import React from 'react';

const fields = [
  { name: 'voltage', label: 'Voltage', placeholder: '1.01' },
  { name: 'frequency', label: 'Frequency', placeholder: '50.0' },
  { name: 'active_power', label: 'Active Power', placeholder: '100.0' },
  { name: 'reactive_power', label: 'Reactive Power', placeholder: '20.0' },
  { name: 'load_demand', label: 'Load Demand', placeholder: '120.0' },
  { name: 'renewable_output', label: 'Renewable Output', placeholder: '30.0' },
  { name: 'temperature', label: 'Temperature', placeholder: '30.0' },
  { name: 'humidity', label: 'Humidity', placeholder: '60.0' },
  { name: 'power_loss', label: 'Power Loss', placeholder: '5.0' },
  { name: 'stability_index', label: 'Stability Index', placeholder: '15.5' },
];

const FeatureInputForm = ({ features, onChange, onSubmit, isSubmitting }) => {
  const isFormValid = fields.every(({ name }) => features[name] !== '' && !Number.isNaN(parseFloat(features[name])));

  return (
    <section className="section">
      <div className="section-heading">
        <div>
          <h2 className="section-title">Grid Measurements</h2>
          <p className="section-description">Enter the ten electrical and environmental features used by the trained OCSVM.</p>
        </div>
        <span className="section-description">10 model inputs</span>
      </div>
      <div className="panel">
        <div className="measurement-grid">
          {fields.map((field) => (
            <label className="field" key={field.name}>
              <span className="field-label">{field.label}</span>
              <input
                className="field-input"
                type="number"
                step="any"
                value={features[field.name]}
                onChange={(e) => onChange(field.name, e.target.value)}
                placeholder={field.placeholder}
                disabled={isSubmitting}
              />
            </label>
          ))}
        </div>
        <div className="form-actions">
          <button className="primary-button" onClick={onSubmit} disabled={!isFormValid || isSubmitting}>
            {isSubmitting ? 'Analyzing Grid Condition…' : '⚡ Detect Fault'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default FeatureInputForm;
