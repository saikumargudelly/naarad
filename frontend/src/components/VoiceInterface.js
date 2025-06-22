import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Settings, 
  Download,
  Upload,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

const VoiceInterface = ({ 
  onVoiceInput, 
  onVoiceOutput, 
  isListening = false, 
  isProcessing = false,
  voiceEnabled = true,
  onToggleVoice,
  className = ""
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [selectedVoice, setSelectedVoice] = useState('alloy');
  const [audioFormat, setAudioFormat] = useState('mp3');
  const [showSettings, setShowSettings] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');
  const [error, setError] = useState(null);
  const [voiceTestResults, setVoiceTestResults] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  // Available voices and formats
  const availableVoices = [
    { id: 'alloy', name: 'Alloy', description: 'Balanced and clear' },
    { id: 'echo', name: 'Echo', description: 'Warm and friendly' },
    { id: 'fable', name: 'Fable', description: 'Storytelling voice' },
    { id: 'onyx', name: 'Onyx', description: 'Deep and authoritative' },
    { id: 'nova', name: 'Nova', description: 'Bright and energetic' },
    { id: 'shimmer', name: 'Shimmer', description: 'Soft and melodic' }
  ];

  const audioFormats = [
    { id: 'mp3', name: 'MP3', description: 'Compressed, widely supported' },
    { id: 'wav', name: 'WAV', description: 'Uncompressed, high quality' },
    { id: 'ogg', name: 'OGG', description: 'Open format, good compression' }
  ];

  // Initialize audio context and permissions
  useEffect(() => {
    const initializeAudio = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          await navigator.mediaDevices.getUserMedia({ audio: true });
        }
      } catch (err) {
        setError('Microphone access denied. Please enable microphone permissions.');
        console.error('Audio initialization error:', err);
      }
    };

    if (voiceEnabled) {
      initializeAudio();
    }
  }, [voiceEnabled]);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setProcessingStatus('Initializing microphone...');
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(audioBlob);
        setAudioUrl(URL.createObjectURL(audioBlob));
        processAudioInput(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setProcessingStatus('Recording...');
      
      if (onVoiceInput) {
        onVoiceInput({ type: 'recording_started' });
      }
    } catch (err) {
      setError('Failed to start recording: ' + err.message);
      setProcessingStatus('');
    }
  }, [onVoiceInput]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      setProcessingStatus('Processing audio...');
      
      if (onVoiceInput) {
        onVoiceInput({ type: 'recording_stopped' });
      }
    }
  }, [isRecording, onVoiceInput]);

  // Process audio input
  const processAudioInput = async (blob) => {
    try {
      setProcessingStatus('Transcribing audio...');
      
      // Convert blob to base64
      const arrayBuffer = await blob.arrayBuffer();
      const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      
      // Send to backend
      const response = await fetch('/api/v1/voice/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio_data: base64Audio,
          user_id: 'current_user', // Replace with actual user ID
          voice_preference: selectedVoice,
          generate_audio: true
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setTranscription(result.transcribed_text);
        setProcessingStatus('Transcription complete');
        
        if (onVoiceInput) {
          onVoiceInput({
            type: 'transcription_complete',
            text: result.transcribed_text,
            audio: result.audio_response
          });
        }
      } else {
        throw new Error(result.error || 'Processing failed');
      }
    } catch (err) {
      setError('Failed to process audio: ' + err.message);
      setProcessingStatus('');
    }
  };

  // Play audio response
  const playAudioResponse = useCallback(async (audioData) => {
    try {
      if (audioRef.current) {
        audioRef.current.pause();
      }

      const audioBlob = new Blob([Uint8Array.from(atob(audioData), c => c.charCodeAt(0))], {
        type: 'audio/mp3'
      });
      
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(audioUrl);
      
      audioRef.current = new Audio(audioUrl);
      audioRef.current.volume = volume;
      
      audioRef.current.onended = () => {
        setIsPlaying(false);
      };
      
      audioRef.current.play();
      setIsPlaying(true);
      
      if (onVoiceOutput) {
        onVoiceOutput({ type: 'audio_playing', audioUrl });
      }
    } catch (err) {
      setError('Failed to play audio: ' + err.message);
    }
  }, [volume, onVoiceOutput]);

  // Synthesize text to speech
  const synthesizeSpeech = async (text) => {
    try {
      setProcessingStatus('Generating speech...');
      
      const formData = new FormData();
      formData.append('text', text);
      formData.append('voice', selectedVoice);
      formData.append('format', audioFormat);
      
      const response = await fetch('/api/v1/voice/synthesize', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        await playAudioResponse(result.audio_data);
        setProcessingStatus('Speech generated');
      } else {
        throw new Error(result.error || 'Synthesis failed');
      }
    } catch (err) {
      setError('Failed to synthesize speech: ' + err.message);
      setProcessingStatus('');
    }
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('audio/')) {
      setUploadedFile(file);
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      processAudioInput(file);
    } else {
      setError('Please select a valid audio file');
    }
  };

  // Test voice features
  const testVoiceFeatures = async (testType) => {
    try {
      setProcessingStatus(`Running ${testType} test...`);
      
      let requestBody = { test_type: testType };
      
      if (testType === 'recognition' && audioBlob) {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
        requestBody.audio_data = base64Audio;
      } else if (testType === 'synthesis') {
        requestBody.text_data = 'This is a test of speech synthesis.';
      } else if (testType === 'full' && audioBlob) {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
        requestBody.audio_data = base64Audio;
      }
      
      const response = await fetch('/api/v1/voice/test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setVoiceTestResults(result);
      setProcessingStatus(`${testType} test completed`);
    } catch (err) {
      setError(`Failed to run ${testType} test: ` + err.message);
      setProcessingStatus('');
    }
  };

  // Health check
  const checkVoiceHealth = async () => {
    try {
      const response = await fetch('/api/v1/voice/health');
      const result = await response.json();
      
      if (result.status === 'healthy') {
        setError(null);
        setProcessingStatus('Voice service is healthy');
      } else {
        setError('Voice service is unhealthy: ' + result.error);
      }
    } catch (err) {
      setError('Failed to check voice health: ' + err.message);
    }
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className={`voice-interface ${className}`}>
      {/* Main Controls */}
      <div className="voice-controls">
        <div className="control-group">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!voiceEnabled || isProcessing}
            className={`record-button ${isRecording ? 'recording' : ''} ${!voiceEnabled ? 'disabled' : ''}`}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            {isRecording ? <MicOff size={24} /> : <Mic size={24} />}
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="settings-button"
            title="Voice Settings"
          >
            <Settings size={20} />
          </button>
        </div>

        <div className="status-indicators">
          {isRecording && (
            <div className="recording-indicator">
              <div className="pulse-dot"></div>
              Recording...
            </div>
          )}
          
          {processingStatus && (
            <div className="processing-status">
              <div className="spinner"></div>
              {processingStatus}
            </div>
          )}
          
          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel">
          <div className="setting-group">
            <label>Voice Selection:</label>
            <select 
              value={selectedVoice} 
              onChange={(e) => setSelectedVoice(e.target.value)}
            >
              {availableVoices.map(voice => (
                <option key={voice.id} value={voice.id}>
                  {voice.name} - {voice.description}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-group">
            <label>Audio Format:</label>
            <select 
              value={audioFormat} 
              onChange={(e) => setAudioFormat(e.target.value)}
            >
              {audioFormats.map(format => (
                <option key={format.id} value={format.id}>
                  {format.name} - {format.description}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-group">
            <label>Volume: {Math.round(volume * 100)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
            />
          </div>

          <div className="setting-actions">
            <button onClick={checkVoiceHealth} className="health-check-btn">
              <CheckCircle size={16} />
              Health Check
            </button>
            
            <button onClick={() => testVoiceFeatures('recognition')} className="test-btn">
              Test Recognition
            </button>
            
            <button onClick={() => testVoiceFeatures('synthesis')} className="test-btn">
              Test Synthesis
            </button>
          </div>
        </div>
      )}

      {/* Audio Playback */}
      {audioUrl && (
        <div className="audio-playback">
          <div className="playback-controls">
            <button
              onClick={() => {
                if (isPlaying) {
                  audioRef.current?.pause();
                  setIsPlaying(false);
                } else {
                  audioRef.current?.play();
                  setIsPlaying(true);
                }
              }}
              className="play-button"
            >
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            
            <button
              onClick={() => {
                if (audioRef.current) {
                  audioRef.current.currentTime = 0;
                  audioRef.current.play();
                  setIsPlaying(true);
                }
              }}
              className="restart-button"
            >
              <RotateCcw size={16} />
            </button>
            
            <button
              onClick={() => {
                const link = document.createElement('a');
                link.href = audioUrl;
                link.download = `voice_${Date.now()}.${audioFormat}`;
                link.click();
              }}
              className="download-button"
            >
              <Download size={16} />
            </button>
          </div>
        </div>
      )}

      {/* File Upload */}
      <div className="file-upload">
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="upload-button"
          title="Upload Audio File"
        >
          <Upload size={20} />
          Upload Audio
        </button>
      </div>

      {/* Transcription Display */}
      {transcription && (
        <div className="transcription-display">
          <h4>Transcription:</h4>
          <p>{transcription}</p>
          <button
            onClick={() => synthesizeSpeech(transcription)}
            className="synthesize-button"
          >
            <Volume2 size={16} />
            Synthesize Response
          </button>
        </div>
      )}

      {/* Test Results */}
      {voiceTestResults && (
        <div className="test-results">
          <h4>Voice Test Results:</h4>
          <pre>{JSON.stringify(voiceTestResults, null, 2)}</pre>
        </div>
      )}

      <style jsx>{`
        .voice-interface {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 20px;
          border: 1px solid #333;
          max-width: 500px;
        }

        .voice-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .control-group {
          display: flex;
          gap: 10px;
          align-items: center;
        }

        .record-button {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          border: none;
          background: #2563eb;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .record-button:hover {
          background: #1d4ed8;
          transform: scale(1.05);
        }

        .record-button.recording {
          background: #dc2626;
          animation: pulse 1.5s infinite;
        }

        .record-button.disabled {
          background: #6b7280;
          cursor: not-allowed;
        }

        .settings-button {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          border: 1px solid #374151;
          background: #374151;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .status-indicators {
          display: flex;
          flex-direction: column;
          gap: 8px;
          font-size: 14px;
        }

        .recording-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #dc2626;
        }

        .pulse-dot {
          width: 8px;
          height: 8px;
          background: #dc2626;
          border-radius: 50%;
          animation: pulse 1s infinite;
        }

        .processing-status {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #f59e0b;
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid #f59e0b;
          border-top: 2px solid transparent;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .error-message {
          display: flex;
          align-items: center;
          gap: 8px;
          color: #ef4444;
        }

        .settings-panel {
          background: #262626;
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 16px;
          border: 1px solid #404040;
        }

        .setting-group {
          margin-bottom: 16px;
        }

        .setting-group label {
          display: block;
          margin-bottom: 8px;
          color: #e5e7eb;
          font-weight: 500;
        }

        .setting-group select,
        .setting-group input[type="range"] {
          width: 100%;
          padding: 8px;
          border-radius: 6px;
          border: 1px solid #404040;
          background: #1f1f1f;
          color: #e5e7eb;
        }

        .setting-actions {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .health-check-btn,
        .test-btn {
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid #404040;
          background: #374151;
          color: white;
          cursor: pointer;
          font-size: 12px;
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .health-check-btn:hover,
        .test-btn:hover {
          background: #4b5563;
        }

        .audio-playback {
          margin: 16px 0;
          padding: 12px;
          background: #262626;
          border-radius: 8px;
          border: 1px solid #404040;
        }

        .playback-controls {
          display: flex;
          gap: 8px;
          align-items: center;
        }

        .play-button,
        .restart-button,
        .download-button {
          padding: 8px;
          border-radius: 6px;
          border: 1px solid #404040;
          background: #374151;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .play-button:hover,
        .restart-button:hover,
        .download-button:hover {
          background: #4b5563;
        }

        .file-upload {
          margin: 16px 0;
        }

        .upload-button {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 2px dashed #404040;
          background: transparent;
          color: #e5e7eb;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: all 0.3s ease;
        }

        .upload-button:hover {
          border-color: #2563eb;
          background: rgba(37, 99, 235, 0.1);
        }

        .transcription-display {
          margin: 16px 0;
          padding: 12px;
          background: #262626;
          border-radius: 8px;
          border: 1px solid #404040;
        }

        .transcription-display h4 {
          margin: 0 0 8px 0;
          color: #e5e7eb;
        }

        .transcription-display p {
          margin: 0 0 12px 0;
          color: #d1d5db;
          line-height: 1.5;
        }

        .synthesize-button {
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid #2563eb;
          background: #2563eb;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 14px;
        }

        .synthesize-button:hover {
          background: #1d4ed8;
        }

        .test-results {
          margin: 16px 0;
          padding: 12px;
          background: #262626;
          border-radius: 8px;
          border: 1px solid #404040;
        }

        .test-results h4 {
          margin: 0 0 8px 0;
          color: #e5e7eb;
        }

        .test-results pre {
          margin: 0;
          color: #d1d5db;
          font-size: 12px;
          overflow-x: auto;
        }

        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default VoiceInterface; 