import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

// Add dark mode class to HTML element
if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.classList.add('dark');
}

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
