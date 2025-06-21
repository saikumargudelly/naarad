import React from 'react';
import { motion } from 'framer-motion';
import { UserCircleIcon } from '@heroicons/react/24/outline';

const Message = ({ message }) => {
  const isUser = message.sender === 'user';
  const isError = message.isError;

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2 }}
        className={`flex items-start space-x-3 max-w-[80%] ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}
      >
        {/* Avatar */}
        <div className={`flex-shrink-0 ${isUser ? 'order-2' : 'order-1'}`}>
          {isUser ? (
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <UserCircleIcon className="w-6 h-6 text-white" />
            </div>
          ) : (
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm font-bold">N</span>
            </div>
          )}
        </div>

        {/* Message Content */}
        <div className={`flex-1 ${isUser ? 'order-1' : 'order-2'}`}>
          <div
            className={`px-4 py-3 rounded-2xl ${
              isUser
                ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
                : isError
                ? 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-800'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
            }`}
          >
            {/* Image if present */}
            {message.image && (
              <div className="mb-3">
                <img
                  src={message.image}
                  alt="User uploaded"
                  className="max-w-full h-auto rounded-lg"
                  style={{ maxHeight: '200px' }}
                />
              </div>
            )}

            {/* Message text */}
            <div className="whitespace-pre-wrap break-words">
              {message.text}
            </div>
          </div>

          {/* Timestamp */}
          <div className={`text-xs text-gray-500 dark:text-gray-400 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
            {formatTime(message.timestamp)}
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Message; 