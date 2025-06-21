import React, { useState, useRef, useCallback } from 'react';
import { PaperAirplaneIcon, PhotoIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';

const MessageInput = ({ onSendMessage, isLoading, isSending }) => {
  const [inputMessage, setInputMessage] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  // Handle image selection
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Remove selected image
  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle sending a new message
  const handleSendMessage = useCallback(() => {
    if ((!inputMessage.trim() && !selectedImage) || isLoading) return;

    // Send message to backend
    onSendMessage(inputMessage);
    
    // Clear input
    setInputMessage('');
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [inputMessage, isLoading, onSendMessage, selectedImage]);

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-3">
      {/* Image Preview */}
      {imagePreview && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="relative inline-block"
        >
          <img
            src={imagePreview}
            alt="Preview"
            className="max-h-32 rounded-lg border border-gray-200 dark:border-gray-700"
          />
          <button
            onClick={removeImage}
            className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        </motion.div>
      )}

      {/* Input Area */}
      <div className="flex items-end space-x-3">
        {/* File Upload Button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="flex-shrink-0 p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Upload image"
        >
          <PhotoIcon className="w-5 h-5" />
        </button>
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="hidden"
        />

        {/* Text Input */}
        <div className="flex-1 relative">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-2xl bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none disabled:opacity-50 disabled:cursor-not-allowed"
            rows="1"
            style={{ minHeight: '48px', maxHeight: '120px' }}
          />
        </div>

        {/* Send Button */}
        <button
          onClick={handleSendMessage}
          disabled={(!inputMessage.trim() && !selectedImage) || isLoading}
          className="flex-shrink-0 p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full hover:from-purple-600 hover:to-pink-600 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
          title="Send message"
        >
          {isSending ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <PaperAirplaneIcon className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Character count and tips */}
      <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
        <span>
          {inputMessage.length}/5000 characters
        </span>
        <span>
          Press Enter to send, Shift+Enter for new line
        </span>
      </div>
    </div>
  );
};

export default MessageInput; 