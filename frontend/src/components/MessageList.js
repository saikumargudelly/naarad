import React from 'react';
import { motion } from 'framer-motion';
import Message from './Message';

const MessageList = ({ messages }) => {
  // Safety check for undefined or null messages
  if (!messages) {
    messages = [];
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-md"
        >
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-white text-2xl">😺</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Welcome to Naarad!
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            I'm your curious AI companion. Ask me anything, and I'll do my best to help you out!
          </p>
          <div className="space-y-2 text-sm text-gray-500 dark:text-gray-500">
            <p>💡 Try asking me about:</p>
            <ul className="space-y-1">
              <li>• Current events and news</li>
              <li>• Technical questions</li>
              <li>• Creative writing</li>
              <li>• Problem solving</li>
            </ul>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <motion.div
          key={`${message.sender}-${index}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <Message message={message} />
        </motion.div>
      ))}
    </div>
  );
};

export default MessageList; 