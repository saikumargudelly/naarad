import React, { useState } from 'react';
import { 
  X, 
  User, 
  BarChart3, 
  Settings, 
  Activity, 
  Clock, 
  TrendingUp,
  Heart,
  Zap,
  Shield,
  Info,
  Mic
} from 'lucide-react';

const Sidebar = ({ userPreferences = {}, analyticsData = null, onClose }) => {
  const [activeTab, setActiveTab] = useState('preferences');

  const tabs = [
    { id: 'preferences', name: 'Preferences', icon: User },
    { id: 'analytics', name: 'Analytics', icon: BarChart3 },
    { id: 'system', name: 'System', icon: Settings }
  ];

  const renderPreferences = () => (
    <div className="preferences-section">
      <h3>User Preferences</h3>
      
      {!userPreferences || Object.keys(userPreferences || {}).length === 0 ? (
        <div className="empty-state">
          <Heart size={48} />
          <p>No preferences learned yet</p>
          <span>Start chatting to build your profile</span>
        </div>
      ) : (
        <div className="preferences-list">
          {Object.entries(userPreferences).map(([key, value]) => (
            <div key={key} className="preference-item">
              <div className="preference-label">{key.replace(/_/g, ' ')}</div>
              <div className="preference-value">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderAnalytics = () => (
    <div className="analytics-section">
      <h3>System Analytics</h3>
      
      {!analyticsData || Object.keys(analyticsData || {}).length === 0 ? (
        <div className="empty-state">
          <BarChart3 size={48} />
          <p>No analytics data available</p>
          <span>Analytics will appear here after usage</span>
        </div>
      ) : (
        <div className="analytics-content">
          {analyticsData.monitoring_summary && (
            <div className="analytics-summary">
              <div className="summary-item">
                <Clock size={16} />
                <span>Time Range: {analyticsData.monitoring_summary.time_range}</span>
              </div>
              <div className="summary-item">
                <Activity size={16} />
                <span>Data Points: {analyticsData.monitoring_summary.data_points}</span>
              </div>
            </div>
          )}
          
          {analyticsData.analysis && (
            <div className="analysis-section">
              <h4>Analysis</h4>
              <p>{analyticsData.analysis}</p>
            </div>
          )}
          
          {analyticsData.chart && (
            <div className="chart-section">
              <h4>Visualization</h4>
              <div className="chart-placeholder">
                <TrendingUp size={32} />
                <span>Chart visualization would appear here</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderSystem = () => (
    <div className="system-section">
      <h3>System Information</h3>
      
      <div className="system-info">
        <div className="info-item">
          <Zap size={16} />
          <span>Real-time Streaming: Enabled</span>
        </div>
        
        <div className="info-item">
          <Mic size={16} />
          <span>Voice Processing: Active</span>
        </div>
        
        <div className="info-item">
          <Shield size={16} />
          <span>Security: Encrypted</span>
        </div>
        
        <div className="info-item">
          <Activity size={16} />
          <span>Performance: Optimal</span>
        </div>
      </div>
      
      <div className="feature-list">
        <h4>Active Features</h4>
        <ul>
          <li>Multi-Agent Architecture</li>
          <li>WebSocket Real-time Chat</li>
          <li>Voice Recognition & Synthesis</li>
          <li>Personalization Learning</li>
          <li>Analytics & Insights</li>
          <li>Image Analysis</li>
        </ul>
      </div>
      
      <div className="health-status">
        <h4>Service Health</h4>
        <div className="health-item">
          <div className="health-indicator healthy"></div>
          <span>Chat Service</span>
        </div>
        <div className="health-item">
          <div className="health-indicator healthy"></div>
          <span>Voice Service</span>
        </div>
        <div className="health-item">
          <div className="health-indicator healthy"></div>
          <span>Analytics Service</span>
        </div>
        <div className="health-item">
          <div className="health-indicator healthy"></div>
          <span>Personalization Service</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>System Panel</h2>
        <button onClick={onClose} className="close-button">
          <X size={20} />
        </button>
      </div>

      <div className="sidebar-tabs">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            >
              <Icon size={16} />
              {tab.name}
            </button>
          );
        })}
      </div>

      <div className="sidebar-content">
        {activeTab === 'preferences' && renderPreferences()}
        {activeTab === 'analytics' && renderAnalytics()}
        {activeTab === 'system' && renderSystem()}
      </div>

      <style>{`
        .sidebar {
          width: 350px;
          background: #1a1a1a;
          border-right: 1px solid #333;
          display: flex;
          flex-direction: column;
          height: 100%;
        }

        .sidebar-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 1px solid #333;
        }

        .sidebar-header h2 {
          margin: 0;
          font-size: 20px;
          font-weight: 600;
          color: #e5e7eb;
        }

        .close-button {
          padding: 8px;
          border-radius: 6px;
          border: 1px solid #374151;
          background: #374151;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .close-button:hover {
          background: #4b5563;
        }

        .sidebar-tabs {
          display: flex;
          border-bottom: 1px solid #333;
        }

        .tab-button {
          flex: 1;
          padding: 12px;
          border: none;
          background: transparent;
          color: #9ca3af;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          font-size: 14px;
          transition: all 0.3s ease;
        }

        .tab-button:hover {
          background: #262626;
          color: #e5e7eb;
        }

        .tab-button.active {
          background: #2563eb;
          color: white;
        }

        .sidebar-content {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
        }

        .preferences-section,
        .analytics-section,
        .system-section {
          color: #e5e7eb;
        }

        .preferences-section h3,
        .analytics-section h3,
        .system-section h3 {
          margin: 0 0 16px 0;
          font-size: 18px;
          font-weight: 600;
        }

        .empty-state {
          text-align: center;
          padding: 40px 20px;
          color: #9ca3af;
        }

        .empty-state svg {
          margin-bottom: 16px;
          opacity: 0.5;
        }

        .empty-state p {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 500;
        }

        .empty-state span {
          font-size: 14px;
          opacity: 0.7;
        }

        .preferences-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .preference-item {
          padding: 12px;
          background: #262626;
          border-radius: 8px;
          border: 1px solid #404040;
        }

        .preference-label {
          font-size: 12px;
          font-weight: 500;
          color: #9ca3af;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 4px;
        }

        .preference-value {
          font-size: 14px;
          color: #e5e7eb;
          word-break: break-word;
        }

        .analytics-content {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .analytics-summary {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .summary-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #d1d5db;
        }

        .analysis-section h4,
        .chart-section h4 {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 500;
        }

        .analysis-section p {
          margin: 0;
          font-size: 14px;
          line-height: 1.5;
          color: #d1d5db;
        }

        .chart-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          padding: 40px 20px;
          background: #262626;
          border-radius: 8px;
          border: 1px solid #404040;
          color: #9ca3af;
        }

        .chart-placeholder span {
          font-size: 14px;
          text-align: center;
        }

        .system-info {
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-bottom: 24px;
        }

        .info-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #d1d5db;
        }

        .feature-list {
          margin-bottom: 24px;
        }

        .feature-list h4 {
          margin: 0 0 12px 0;
          font-size: 16px;
          font-weight: 500;
        }

        .feature-list ul {
          margin: 0;
          padding-left: 20px;
          color: #d1d5db;
        }

        .feature-list li {
          margin-bottom: 6px;
          font-size: 14px;
        }

        .health-status h4 {
          margin: 0 0 12px 0;
          font-size: 16px;
          font-weight: 500;
        }

        .health-item {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
          font-size: 14px;
          color: #d1d5db;
        }

        .health-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .health-indicator.healthy {
          background: #10b981;
        }

        .health-indicator.warning {
          background: #f59e0b;
        }

        .health-indicator.error {
          background: #ef4444;
        }

        @media (max-width: 768px) {
          .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            width: 100%;
            z-index: 1000;
          }
        }
      `}</style>
    </div>
  );
};

export default Sidebar; 