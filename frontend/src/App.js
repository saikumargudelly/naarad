import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { PaperAirplaneIcon, PhotoIcon } from '@heroicons/react/24/outline';

function App() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm Naarad, your cat-like AI companion. How can I help you today? 😺",
      sender: 'bot',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() && !selectedImage) return;

    // Add user message to chat
    const userMessage = {
      id: messages.length + 1,
      text: message,
      sender: 'user',
      timestamp: new Date().toISOString(),
      image: selectedImage ? URL.createObjectURL(selectedImage) : null,
    };

    setMessages((prev) => [...prev, userMessage]);
    setMessage('');
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('message', message);
      if (selectedImage) {
        formData.append('image', selectedImage);
      }

      // Get the last few messages for context
      const recentMessages = messages
        .filter((msg) => msg.sender === 'user' || msg.sender === 'bot')
        .slice(-5)
        .map((msg) => ({
          role: msg.sender === 'user' ? 'user' : 'assistant',
          content: msg.text,
        }));

      formData.append('chat_history', JSON.stringify(recentMessages));

      const response = await axios.post('/api/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Add bot response to chat
      const botMessage = {
        id: messages.length + 2,
        text: response.data.message,
        sender: 'bot',
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        id: messages.length + 2,
        text: 'Sorry, I encountered an error. Please try again later.',
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true,
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.match('image.*')) {
      alert('Please select an image file');
      return;
    }

    // Check file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      alert('Image size should be less than 5MB');
      return;
    }

    setSelectedImage(file);
    
    // Create preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center">
          <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center mr-3">
            <span className="text-2xl">😺</span>
          </div>
          <h1 className="text-xl font-bold">Naarad AI</h1>
        </div>
      </header>

      {/* Chat container */}
      <div className="flex-1 overflow-y-auto p-4 pb-24">
        <div className="container mx-auto max-w-4xl">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex mb-4 ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`rounded-lg p-4 max-w-3xl ${
                  msg.sender === 'user'
                    ? 'bg-indigo-500 text-white rounded-br-none'
                    : 'bg-white border border-gray-200 rounded-bl-none shadow-sm'
                }`}
              >
                {msg.image && (
                  <div className="mb-2">
                    <img
                      src={msg.image}
                      alt="User uploaded"
                      className="max-h-48 rounded"
                    />
                  </div>
                )}
                <p className="whitespace-pre-wrap">{msg.text}</p>
                <p className="text-xs mt-1 opacity-70">
                  {new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="bg-white border border-gray-200 rounded-lg rounded-bl-none p-4 max-w-3xl">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4 shadow-lg">
        <div className="container mx-auto max-w-4xl">
          {imagePreview && (
            <div className="relative mb-3 max-w-xs">
              <img
                src={imagePreview}
                alt="Preview"
                className="h-24 rounded border border-gray-300"
              />
              <button
                onClick={removeImage}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm hover:bg-red-600"
              >
                ×
              </button>
            </div>
          )}
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Message Naarad..."
                className="w-full p-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                disabled={isLoading}
              />
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageChange}
                accept="image/*"
                className="hidden"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="absolute right-14 top-2.5 text-gray-500 hover:text-indigo-600 p-1 rounded-full"
                disabled={isLoading}
              >
                <PhotoIcon className="h-6 w-6" />
              </button>
            </div>
            <button
              type="submit"
              disabled={isLoading || (!message.trim() && !selectedImage)}
              className="bg-indigo-600 text-white p-3 rounded-full hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PaperAirplaneIcon className="h-5 w-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
