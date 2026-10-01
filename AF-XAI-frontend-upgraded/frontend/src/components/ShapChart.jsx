import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

const labels = {
  voltage: 'Voltage', frequency: 'Frequency', active_power: 'Active Power',
  reactive_power: 'Reactive Power', load_demand: 'Load Demand', renewable_output: 'Renewable Output',
  temperature: 'Temperature', humidity: 'Humidity', power_loss: 'Power Loss', stability_index: 'Stability Index',
};

const ShapChart = ({ explanations }) => {
  if (!explanations?.length) return null;
  const chartData = explanations
    .map(item => ({ feature: labels[item.feature] || item.feature, value: item.shap_value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="panel chart-panel">
      <div className="section-heading">
        <div>
          <h2 className="section-title">Explainable AI — SHAP Contributions</h2>
          <p className="section-description">Feature contributions to the model anomaly score for this prediction.</p>
        </div>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 28, left: 95, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" width={95} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value) => [Number(value).toFixed(4), 'SHAP value']} />
            <Bar dataKey="value" radius={[0, 5, 5, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.value >= 0 ? '#c94c43' : '#3f6fa8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <div className="legend-item"><span className="legend-dot" style={{ background:'#c94c43' }} />Positive contribution → pushes anomaly score upward</div>
        <div className="legend-item"><span className="legend-dot" style={{ background:'#3f6fa8' }} />Negative contribution → pushes anomaly score downward</div>
      </div>
    </div>
  );
};

export default ShapChart;
