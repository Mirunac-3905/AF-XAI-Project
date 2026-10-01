import React, { useCallback, useState } from 'react';
import { predictFault } from '../services/api';
import Header from '../components/Header.jsx';
import ApiStatus from '../components/ApiStatus.jsx';
import FeatureInputForm from '../components/FeatureInputForm.jsx';
import PredictionCard from '../components/PredictionCard.jsx';
import ScoreCard from '../components/ScoreCard.jsx';
import ShapChart from '../components/ShapChart.jsx';
import InterpretationCard from '../components/InterpretationCard.jsx';

const EMPTY_FEATURES = {
  voltage: '', frequency: '', active_power: '', reactive_power: '', load_demand: '',
  renewable_output: '', temperature: '', humidity: '', power_loss: '', stability_index: '',
};

const clients = [
  { name: 'Client 1', weight: 33.1815 },
  { name: 'Client 2', weight: 33.6237 },
  { name: 'Client 3', weight: 33.1949 },
];

const Dashboard = () => {
  const [backendConnected, setBackendConnected] = useState(false);
  const [features, setFeatures] = useState(EMPTY_FEATURES);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleStatusChange = useCallback((connected) => setBackendConnected(connected), []);
  const handleFeatureChange = (name, value) => {
    setFeatures(prev => ({ ...prev, [name]: value }));
    if (error) setError(null);
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    try {
      const numericalFeatures = Object.fromEntries(Object.entries(features).map(([key, value]) => [key, parseFloat(value)]));
      const response = await predictFault(numericalFeatures);
      if (response.success && response.prediction) setPrediction(response.prediction);
      else setError('Invalid response from prediction API');
    } catch (err) {
      setPrediction(null);
      setError(err.message || 'Failed to get prediction');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setFeatures(EMPTY_FEATURES);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="dashboard-shell">
      <main className="dashboard-container">
        <Header backendConnected={backendConnected} />
        <ApiStatus onStatusChange={handleStatusChange} />

        <section className="section">
          <div className="section-heading">
            <div>
              <h2 className="section-title">Federated AI System</h2>
              <p className="section-description">Three simulated substations contribute locally without sharing raw measurements.</p>
            </div>
          </div>
          <div className="status-grid">
            <div className="status-card">
              <div className="status-card-top"><span className="status-card-label">SUBSTATIONS</span><span className="status-icon">3</span></div>
              <div className="status-card-value">3 Clients</div>
              <div className="status-card-sub">Distributed local OCSVM models</div>
            </div>
            <div className="status-card">
              <div className="status-card-top"><span className="status-card-label">GLOBAL MODEL</span><span className="status-icon">AI</span></div>
              <div className="status-card-value">Adaptive OCSVM</div>
              <div className="status-card-sub">915 combined support vectors</div>
            </div>
            <div className="status-card">
              <div className="status-card-top"><span className="status-card-label">EXPLAINABILITY</span><span className="status-icon">X</span></div>
              <div className="status-card-value">SHAP Enabled</div>
              <div className="status-card-sub">Feature-level anomaly explanations</div>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <div>
              <h2 className="section-title">Adaptive Aggregation</h2>
              <p className="section-description">Current quality-weighted contribution of each federated client.</p>
            </div>
          </div>
          <div className="client-grid">
            {clients.map(client => (
              <div className="client-card" key={client.name}>
                <div className="client-top"><span className="client-name">{client.name}</span><span className="client-live">● Active</span></div>
                <div className="weight-row"><span className="weight">{client.weight.toFixed(2)}%</span><span className="weight-label">Aggregation weight</span></div>
                <div className="weight-track"><div className="weight-fill" style={{ width: `${client.weight}%` }} /></div>
              </div>
            ))}
          </div>
        </section>

        <FeatureInputForm features={features} onChange={handleFeatureChange} onSubmit={handleSubmit} isSubmitting={isSubmitting} />

        {error && <div className="error">{error}</div>}

        {prediction && (
          <section className="section">
            <div className="section-heading">
              <div>
                <h2 className="section-title">Detection &amp; Explanation</h2>
                <p className="section-description">Prediction generated by the live Flask API using the adaptive aggregated model.</p>
              </div>
            </div>
            <div className="result-grid">
              <PredictionCard prediction={prediction} />
              <ScoreCard prediction={prediction} />
            </div>
            <ShapChart explanations={prediction.explanations} />
            <div style={{ marginTop: 14 }}><InterpretationCard explanations={prediction.explanations} /></div>
          </section>
        )}

        {(prediction || error) && (
          <div className="reset-row"><button className="secondary-button" onClick={handleReset}>Reset Dashboard</button></div>
        )}

        <div className="footer-note">AF-XAI · Adaptive Federated Explainable AI for Smart Grid Fault Detection · Research Prototype</div>
      </main>
    </div>
  );
};

export default Dashboard;
