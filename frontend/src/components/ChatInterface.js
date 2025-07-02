import React, { useState, useRef, useEffect, useCallback } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import VoiceInterface from './VoiceInterface';
import Sidebar from './Sidebar';
import { 
  Mic, 
  MicOff, 
  Settings, 
  Volume2, 
  VolumeX,
  MessageSquare,
  Zap,
  BarChart3,
  User
} from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [showVoiceInterface, setShowVoiceInterface] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [userPreferences, setUserPreferences] = useState({});
  const [analyticsData, setAnalyticsData] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [currentUserId] = useState(`user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [currentConversationId] = useState(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const wsRef = useRef(null);

  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Establish a single, stable WebSocket connection.
  useEffect(() => {
    // If a connection already exists, do nothing.
    if (wsRef.current) return;

    // Use the correct backend endpoint for WebSocket chat
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/chat/${currentUserId}/${currentConversationId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = (event) => {
      setConnectionStatus('disconnected');
      console.log('WebSocket closed:', event.code, event.reason);
    };

    ws.onerror = (error) => {
      setConnectionStatus('error');
      console.error('WebSocket error:', error);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
        wsRef.current = null;
      }
    };
  }, []); // Empty dependency array ensures this runs only once.

  // Handle WebSocket messages
  const handleWebSocketMessage = (data) => {
    // Always remove typing indicator when a new message comes in
    setMessages(prev => prev.filter(msg => !msg.isTyping));

    switch (data.type) {
      case 'message':
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: data.content,
          sender: 'assistant',
          timestamp: new Date().toISOString(),
          isStreaming: false
        }]);
        setIsLoading(false);
        break;
      
      case 'stream_chunk':
        setMessages(prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          
          if (lastMessage && lastMessage.sender === 'assistant' && lastMessage.isStreaming) {
            lastMessage.text += data.chunk;
          } else {
            newMessages.push({
              id: Date.now(),
              text: data.chunk,
              sender: 'assistant',
              timestamp: new Date().toISOString(),
              isStreaming: true
            });
          }
          
          return newMessages;
        });
        break;
      
      case 'stream_end':
        setMessages(prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          
          if (lastMessage && lastMessage.sender === 'assistant') {
            lastMessage.isStreaming = false;
          }
          
          return newMessages;
        });
        setIsLoading(false);
        break;
      
      case 'typing_start':
        // Handle typing indicator
        setMessages(prev => {
          // Prevent adding multiple typing indicators
          if (prev.some(msg => msg.isTyping)) {
            return prev;
          }
          return [...prev, {
            id: Date.now(),
            text: '🤖 Assistant is typing...',
            sender: 'system',
            timestamp: new Date().toISOString(),
            isTyping: true
          }];
        });
        break;
      
      case 'typing_stop':
        // Remove typing indicator
        setMessages(prev => prev.filter(msg => !msg.isTyping));
        break;
      
      case 'conversation_id':
        // Store conversation ID for future reference
        if (data.conversation_id) {
          console.log('Conversation ID:', data.conversation_id);
        }
        break;
      
      case 'message_complete':
        // Handle message completion
        setIsLoading(false);
        // Remove typing indicator if present
        setMessages(prev => prev.filter(msg => !msg.isTyping));
        break;
      
      case 'error':
        console.error('WebSocket error from backend:', data.error || 'No error message provided');
        setIsLoading(false);
        break;
      
      default:
        // Log unknown message types but don't show error
        if (data.type && !['ping', 'pong', 'heartbeat'].includes(data.type)) {
          console.log('Unknown message type:', data.type, data);
        }
    }
  };

  // Only send message on user action
  const sendMessage = (message, conversationId = null) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const messageData = {
        type: 'text',
        message,
        conversation_id: conversationId,
        timestamp: new Date().toISOString(),
      };
      wsRef.current.send(JSON.stringify(messageData));
      setIsLoading(true);
      // Optionally, add the user message to the UI immediately
      setMessages(prev => [
        ...prev,
        {
          id: Date.now(),
          text: message,
          sender: 'user',
          timestamp: new Date().toISOString(),
        },
      ]);
    } else {
      alert('Connection lost. Please refresh the page.');
    }
  };

  // Handle voice input
  const handleVoiceInput = useCallback(async (voiceData) => {
    switch (voiceData.type) {
      case 'recording_started':
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: '🎤 Recording...',
          sender: 'system',
          timestamp: new Date().toISOString()
        }]);
        break;
      
      case 'recording_stopped':
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: '🔄 Processing audio...',
          sender: 'system',
          timestamp: new Date().toISOString()
        }]);
        break;
      
      case 'transcription_complete':
        // Remove the processing message
        setMessages(prev => prev.filter(msg => msg.text !== '🔄 Processing audio...'));
        
        // Add transcription message
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: `🎤 "${voiceData.text}"`,
          sender: 'user',
          timestamp: new Date().toISOString(),
          isVoiceInput: true
        }]);
        
        // Send the transcribed text
        sendMessage(voiceData.text, null);
        break;
    }
  }, [sendMessage]);

  // Handle voice output
  const handleVoiceOutput = useCallback((voiceData) => {
    if (voiceData.type === 'audio_playing') {
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: '🔊 Playing audio response...',
        sender: 'system',
        timestamp: new Date().toISOString()
      }]);
    }
  }, []);

  // Learn from interaction for personalization
  const learnFromInteraction = async (userMessage, response) => {
    try {
      await fetch('/api/v1/personalization/learn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: currentUserId,
          message: userMessage,
          response: response,
          interaction_type: 'chat',
          context: {
            timestamp: new Date().toISOString(),
            source: 'web_interface'
          }
        })
      });
    } catch (error) {
      console.error('Failed to learn from interaction:', error);
    }
  };

  // Get user preferences
  const fetchUserPreferences = async () => {
    try {
      const response = await fetch(`/api/v1/personalization/preferences/${currentUserId}`);
      if (response.ok) {
        const data = await response.json();
        setUserPreferences(data.preferences || {});
      }
    } catch (error) {
      console.error('Failed to fetch user preferences:', error);
    }
  };

  // Get analytics data
  const fetchAnalyticsData = async () => {
    try {
      const response = await fetch('/api/v1/analytics/monitoring', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          time_range: '24h',
          metrics: ['response_time', 'success_rate', 'user_engagement'],
          user_id: currentUserId
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
      }
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    }
  };

  // Test voice features
  const testVoiceFeatures = async () => {
    try {
      const response = await fetch('/api/v1/voice/health');
      if (response.ok) {
        const data = await response.json();
        console.log('Voice service health:', data);
      }
    } catch (error) {
      console.error('Voice health check failed:', error);
    }
  };

  // Load initial data
  useEffect(() => {
    fetchUserPreferences();
    fetchAnalyticsData();
    testVoiceFeatures();
  }, []);

  return (
    <div className="chat-interface">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="sidebar-toggle"
          >
            <Settings size={20} />
          </button>
          
          <h1>Naarad AI Assistant</h1>
          
          <div className="connection-status">
            <div className={`status-indicator ${connectionStatus}`}></div>
            <span>{connectionStatus}</span>
          </div>
        </div>
        
        <div className="header-right">
          <button
            onClick={() => setShowVoiceInterface(!showVoiceInterface)}
            className={`voice-toggle ${voiceEnabled ? 'enabled' : 'disabled'}`}
            title={voiceEnabled ? 'Voice Enabled' : 'Voice Disabled'}
          >
            {voiceEnabled ? <Mic size={20} /> : <MicOff size={20} />}
          </button>
          
          <button
            onClick={fetchAnalyticsData}
            className="analytics-button"
            title="View Analytics"
          >
            <BarChart3 size={20} />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="chat-content">
        {/* Sidebar */}
        {showSidebar && (
          <Sidebar
            userPreferences={userPreferences}
            analyticsData={analyticsData}
            onClose={() => setShowSidebar(false)}
          />
        )}

        {/* Chat Area */}
        <div className="chat-area flex-1 overflow-y-auto">
          <MessageList 
            messages={messages} 
            isLoading={isLoading}
          />
          
          <div ref={messagesEndRef} />
        </div>

        {/* Voice Interface */}
        {showVoiceInterface && (
          <div className="voice-panel">
            <VoiceInterface
              onVoiceInput={handleVoiceInput}
              onVoiceOutput={handleVoiceOutput}
              isListening={isLoading}
              isProcessing={isLoading}
              voiceEnabled={voiceEnabled}
              onToggleVoice={() => setVoiceEnabled(!voiceEnabled)}
              userId={currentUserId}
              conversationId={currentConversationId}
            />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <MessageInput
          onSendMessage={sendMessage}
          isLoading={isLoading}
          voiceEnabled={voiceEnabled}
          onVoiceToggle={() => setVoiceEnabled(!voiceEnabled)}
        />
      </div>

      <style>{`
        .chat-interface {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: #0f0f0f;
          color: #e5e7eb;
        }

        .chat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 24px;
          background: #1a1a1a;
          border-bottom: 1px solid #333;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .header-left {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .header-left h1 {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
          background: linear-gradient(135deg, #2563eb, #7c3aed);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
        }

        .status-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-indicator.connected {
          background: #10b981;
        }

        .status-indicator.disconnected {
          background: #6b7280;
        }

        .status-indicator.error {
          background: #ef4444;
        }

        .header-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .sidebar-toggle,
        .voice-toggle,
        .analytics-button {
          padding: 8px;
          border-radius: 8px;
          border: 1px solid #374151;
          background: #374151;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
        }

        .sidebar-toggle:hover,
        .voice-toggle:hover,
        .analytics-button:hover {
          background: #4b5563;
          transform: translateY(-1px);
        }

        .voice-toggle.enabled {
          background: #2563eb;
          border-color: #2563eb;
        }

        .voice-toggle.enabled:hover {
          background: #1d4ed8;
        }

        .voice-toggle.disabled {
          background: #6b7280;
          border-color: #6b7280;
        }

        .chat-content {
          display: flex;
          flex: 1;
          overflow: hidden;
        }

        .chat-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow-y: auto;
        }

        .voice-panel {
          width: 400px;
          background: #1a1a1a;
          border-left: 1px solid #333;
          overflow-y: auto;
        }

        .input-area {
          padding: 16px 24px;
          background: #1a1a1a;
          border-top: 1px solid #333;
        }

        @media (max-width: 768px) {
          .voice-panel {
            position: fixed;
            top: 0;
            right: 0;
            bottom: 0;
            width: 100%;
            z-index: 1000;
          }
          
          .header-left h1 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatInterface; 