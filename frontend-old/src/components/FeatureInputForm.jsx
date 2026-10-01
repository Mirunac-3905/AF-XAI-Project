import React from 'react';

const FeatureInputForm = ({ features, onChange, onSubmit, isSubmitting }) => {
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

  const handleChange = (name, value) => {
    onChange(name, value);
  };

  const isFormValid = () => {
    return fields.every(field => {
      const value = features[field.name];
      return value !== '' && !isNaN(parseFloat(value));
    });
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.sectionTitle}>Grid Measurements</h2>
      <div style={styles.grid}>
        {fields.map((field) => (
          <div key={field.name} style={styles.fieldContainer}>
            <label style={styles.label}>{field.label}</label>
            <input
              type="number"
              step="any"
              name={field.name}
              value={features[field.name]}
              onChange={(e) => handleChange(field.name, e.target.value)}
              placeholder={field.placeholder}
              style={styles.input}
              disabled={isSubmitting}
            />
          </div>
        ))}
      </div>
      <div style={styles.buttonContainer}>
        <button
          onClick={onSubmit}
          disabled={!isFormValid() || isSubmitting}
          style={{
            ...styles.button,
            backgroundColor: isFormValid() && !isSubmitting ? '#2563eb' : '#9ca3af',
            cursor: isFormValid() && !isSubmitting ? 'pointer' : 'not-allowed',
          }}
        >
          {isSubmitting ? 'Analyzing grid condition...' : 'Detect Fault'}
        </button>
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
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '20px',
    marginBottom: '20px',
  },
  fieldContainer: {
    display: 'flex',
    flexDirection: 'column',
  },
  label: {
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151',
    marginBottom: '6px',
  },
  input: {
    padding: '10px',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    color: '#1f2937',
  },
  buttonContainer: {
    display: 'flex',
    justifyContent: 'center',
    marginTop: '20px',
  },
  button: {
    padding: '12px 40px',
    fontSize: '16px',
    fontWeight: '600',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    transition: 'background-color 0.2s',
  },
};

export default FeatureInputForm;
