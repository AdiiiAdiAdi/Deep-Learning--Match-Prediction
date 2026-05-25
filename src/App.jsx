import React, { useState } from 'react';
import { Activity } from 'lucide-react';
import MatchInsights from './components/Predictor';
import DatasetOverview from './components/DatasetOverview';
import ModelPerformance from './components/ModelPerformance';
import FeatureImportance from './components/FeatureImportance';
import OngoingMatch from './components/OngoingMatch';
import ApiStatusBar from './components/ApiStatusBar';
import { PredictionProvider } from './context/PredictionContext';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('Predictor');

  const tabs = [
    { id: 'Predictor', label: 'Match Insights' },
    { id: 'OngoingMatch', label: 'Live Predict' },
    { id: 'DatasetOverview', label: 'Dataset Overview' },
    { id: 'ModelPerformance', label: 'Model Performance' },
    { id: 'FeatureImportance', label: 'Feature Importance & SHAP' }
  ];

  const renderContent = () => {
    switch(activeTab) {
      case 'Predictor': return <MatchInsights />;
      case 'OngoingMatch': return <OngoingMatch />;
      case 'DatasetOverview': return <DatasetOverview />;
      case 'ModelPerformance': return <ModelPerformance />;
      case 'FeatureImportance': return <FeatureImportance />;
      default: return <MatchInsights />;
    }
  };

  return (
    <PredictionProvider>
      <ApiStatusBar />
      <div className="app-container" style={{ paddingTop: 'calc(1rem + 28px)' }}>
        <header className="app-header">
          <div className="app-title">
            <Activity size={24} color="#39d353" />
            <span>Premier League Deep Learning Outcome Predictor</span>
          </div>
          <nav className="tab-nav">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </header>

        <main className="tab-content" key={activeTab}>
          {renderContent()}
        </main>
      </div>
    </PredictionProvider>
  );
}

export default App;
