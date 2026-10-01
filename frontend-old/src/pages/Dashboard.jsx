import React, { useState } from 'react';
import { predictFault } from '../services/api';
import Header from '../components/Header.jsx';
import ApiStatus from '../components/ApiStatus.jsx';
import FeatureInputForm from '../components/FeatureInputForm.jsx';
import PredictionCard from '../components/PredictionCard.jsx';
import ScoreCard from '../components/ScoreCard.jsx';
import ShapChart from '../components/ShapChart.jsx';
import InterpretationCard from '../components/InterpretationCard.jsx';

const Dashboard = () => {
  const [backendConnected, setBackendConnected] = useState(false);
  const [features, setFeatures] = useState({
    voltage: '',
    frequency: '',
    active_power: '',
    reactive_power: '',
    load_demand: '',
    renewable_output: '',
    temperature: '',
    humidity: '',
    power_loss: '',
    stability_index: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleFeatureChange = (name, value) => {
    setFeatures(prev => ({
      ...prev,
      [name]: value,
    }));
    // Clear error when user starts typing
    if (error) {
      setError(null);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    setPrediction(null);

    try {
      const numericalFeatures = {};
      Object.keys(features).forEach(key => {
        numericalFeatures[key] = parseFloat(features[key]);
      });

      const response = await predictFault(numericalFeatures);

      if (response.success && response.prediction) {
        setPrediction(response.prediction);
      } else {
        setError('Invalid response from backend');
      }
    } catch (err) {
      setError(err.message || 'Failed to get prediction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setFeatures({
      voltage: '',
      frequency: '',
      active_power: '',
      reactive_power: '',
      load_demand: '',
      renewable_output: '',
      temperature: '',
      humidity: '',
      power_loss: '',
      stability_index: '',
    });
    setPrediction(null);
    setError(null);
  };

  return (
    <div style={styles.container}>
      <Header backendConnected={backendConnected} />
      <ApiStatus onStatusChange={setBackendConnected} />

      <div style={styles.mainContent}>
        <FeatureInputForm
          features={features}
          onChange={handleFeatureChange}
          onSubmit={handleSubmit}
          isSubmitting={isSubmitting}
        />

        {error && (
          <div style={styles.errorContainer}>
            <div style={styles.errorText}>{error}</div>
          </div>
        )}

        {prediction && (
          <>
            <PredictionCard prediction={prediction} />
            <ScoreCard prediction={prediction} />
            <ShapChart explanations={prediction.explanations} />
            <InterpretationCard explanations={prediction.explanations} />
          </>
        )}

        {(prediction || error) && (
          <div style={styles.resetContainer}>
            <button onClick={handleReset} style={styles.resetButton}>
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '40px 20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  mainContent: {
    marginTop: '20px',
  },
  errorContainer: {
    padding: '16px',
    backgroundColor: '#fef2f2',
    borderRadius: '8px',
    border: '1px solid #fecaca',
    marginBottom: '20px',
  },
  errorText: {
    color: '#dc2626',
    fontSize: '14px',
    margin: 0,
  },
  resetContainer: {
    display: 'flex',
    justifyContent: 'center',
    marginTop: '30px',
    marginBottom: '20px',
  },
  resetButton: {
    padding: '10px 30px',
    fontSize: '14px',
    fontWeight: '500',
    color: '#374151',
    backgroundColor: '#f3f4f6',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
};

export default Dashboard;
