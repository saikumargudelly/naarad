import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatInterface = ({ 
  messages, 
  setMessages, 
  conversationId, 
  setConversationId,
  isLoading,
  isSending 
}) => {
  const messagesEndRef = useRef(null);
  
  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // API base URL - FIXED: Added /v1 to match backend endpoint
  const API_BASE_URL = 'http://localhost:8000/api/v1';

  // Function to send message to the backend
  const sendMessage = useCallback(async (newMessage) => {
    // 1. Immediately add user message to the UI
    const userMessage = {
      text: newMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      // 2. Prepare the request for the backend
      const chatHistoryForAPI = messages
        .filter(msg => msg.sender === 'user' || msg.sender === 'ai')
        .map(msg => ({
          role: msg.sender === 'user' ? 'user' : 'assistant',
          content: msg.text
        }));

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: newMessage,
          conversation_id: conversationId,
          chat_history: chatHistoryForAPI // Send the structured history
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // 3. Extract response text and add AI message to UI
      const responseText = data.message || JSON.stringify(data, null, 2);

      const aiMessage = {
        text: responseText,
        sender: 'ai',
        timestamp: new Date().toISOString()
      };
      
      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
      }
      
      // Update messages with the AI response
      setMessages(prev => [...prev, aiMessage]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        text: `Sorry, I encountered an error: ${error.message}. Please try again.`,
        sender: 'ai',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [conversationId, messages, setConversationId, setMessages]);

  // Handle new chat
  const handleNewChat = useCallback(() => {
    setMessages([]);
    setConversationId(null);
  }, [setMessages, setConversationId]);

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
            <span className="text-white text-sm font-bold">N</span>
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Naarad AI Assistant
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Your curious AI companion
            </p>
          </div>
        </div>
        
        <button
          onClick={handleNewChat}
          className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          <ArrowPathIcon className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <MessageList messages={messages} />
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4">
        <MessageInput 
          onSendMessage={sendMessage}
          isLoading={isLoading}
          isSending={isSending}
        />
      </div>
    </div>
  );
};

export default ChatInterface; 