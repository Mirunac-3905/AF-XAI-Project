import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ShapChart = ({ explanations }) => {
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

  const chartData = explanations
    .map(item => ({
      feature: featureLabels[item.feature] || item.feature,
      value: item.shap_value,
      rawFeature: item.feature,
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const maxAbsValue = Math.max(...chartData.map(d => Math.abs(d.value)));
  const barWidth = maxAbsValue > 0 ? 400 / maxAbsValue : 0;

  return (
    <div style={styles.container}>
      <h2 style={styles.sectionTitle}>Model Interpretation</h2>
      <div style={styles.chartContainer}>
        <h3 style={styles.chartTitle}>SHAP Feature Contributions</h3>
        <div style={styles.chartWrapper}>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" width={100} />
              <Tooltip
                formatter={(value) => [value.toFixed(4), 'SHAP Value']}
                contentStyle={{
                  backgroundColor: '#f9fafb',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.value >= 0 ? '#ef4444' : '#3b82f6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={styles.legend}>
          <div style={styles.legendItem}>
            <span style={{ ...styles.legendColor, backgroundColor: '#ef4444' }} />
            <span style={styles.legendText}>Positive contribution → pushes toward fault/anomaly</span>
          </div>
          <div style={styles.legendItem}>
            <span style={{ ...styles.legendColor, backgroundColor: '#3b82f6' }} />
            <span style={styles.legendText}>Negative contribution → pushes toward normal</span>
          </div>
        </div>
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
  chartContainer: {
    backgroundColor: '#f9fafb',
    padding: '20px',
    borderRadius: '8px',
    border: '1px solid #e5e7eb',
  },
  chartTitle: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '15px',
  },
  chartWrapper: {
    height: '400px',
  },
  legend: {
    marginTop: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  legendColor: {
    width: '16px',
    height: '16px',
    borderRadius: '4px',
  },
  legendText: {
    fontSize: '13px',
    color: '#6b7280',
  },
};

export default ShapChart;
