import React from 'react';

const Header = ({ backendConnected }) => (
  <header className="header">
    <div>
      <div className="brand-row">
        <div className="brand-mark">AI</div>
        <div>
          <h1 className="title">AF-XAI</h1>
          <p className="subtitle">Adaptive Federated Explainable AI</p>
        </div>
      </div>
      <p className="subtitle" style={{ marginLeft: 58, marginTop: 6 }}>
        Smart Grid Fault Detection &amp; Explainability Dashboard
      </p>
    </div>
    <div className="header-status">
      <span
        className="status-dot"
        style={{ background: backendConnected ? '#22a866' : '#d74a43' }}
      />
      {backendConnected ? 'System Online' : 'Backend Offline'}
    </div>
  </header>
);

export default Header;
