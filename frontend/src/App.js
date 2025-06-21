import React, { useState, useEffect } from 'react';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';

// Import modular components
import { 
  ChatInterface, 
  Sidebar 
} from './components';

function App() {
  // State management
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [darkMode, setDarkMode] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Handle sending message
  const handleSendMessage = async (message) => {
    setIsLoading(true);
    setIsSending(true);
    
    // Add user message to chat
    const userMessage = {
      text: message,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Reset sending state after a small delay for better UX
    setTimeout(() => {
      setIsSending(false);
      setIsLoading(false);
    }, 300);
  };

  // Render content based on active tab
  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <ChatInterface
            messages={messages}
            setMessages={setMessages}
            conversationId={conversationId}
            setConversationId={setConversationId}
            isLoading={isLoading}
            isSending={isSending}
          />
        );
      case 'dashboard':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Dashboard
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Dashboard features coming soon...
              </p>
            </div>
          </div>
        );
      case 'analytics':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Analytics
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Analytics features coming soon...
              </p>
            </div>
          </div>
        );
      case 'settings':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Settings
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Settings features coming soon...
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100 dark:bg-gray-900">
          <div className="container mx-auto px-6 py-8 h-full">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
                className="h-full"
              >
                {renderContent()}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
