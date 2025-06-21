import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Message from '../Message';

describe('Message Component', () => {
  const mockUserMessage = {
    text: 'Hello, how are you?',
    sender: 'user',
    timestamp: '2024-01-15T10:30:00.000Z'
  };

  const mockAIMessage = {
    text: 'I\'m doing great, thank you for asking!',
    sender: 'ai',
    timestamp: '2024-01-15T10:30:01.000Z'
  };

  const mockErrorMessage = {
    text: 'Sorry, I encountered an error.',
    sender: 'ai',
    timestamp: '2024-01-15T10:30:02.000Z',
    isError: true
  };

  const mockMessageWithImage = {
    text: 'Check out this image!',
    sender: 'user',
    timestamp: '2024-01-15T10:30:03.000Z',
    image: 'data:image/jpeg;base64,test-image-data'
  };

  test('renders user message correctly', () => {
    render(<Message message={mockUserMessage} />);
    
    expect(screen.getByText('Hello, how are you?')).toBeInTheDocument();
    expect(screen.getByText('10:30')).toBeInTheDocument();
  });

  test('renders AI message correctly', () => {
    render(<Message message={mockAIMessage} />);
    
    expect(screen.getByText('I\'m doing great, thank you for asking!')).toBeInTheDocument();
    expect(screen.getByText('10:30')).toBeInTheDocument();
  });

  test('renders error message correctly', () => {
    render(<Message message={mockErrorMessage} />);
    
    expect(screen.getByText('Sorry, I encountered an error.')).toBeInTheDocument();
  });

  test('renders message with image', () => {
    render(<Message message={mockMessageWithImage} />);
    
    expect(screen.getByText('Check out this image!')).toBeInTheDocument();
    expect(screen.getByAltText('User uploaded')).toBeInTheDocument();
  });

  test('formats timestamp correctly', () => {
    const messageWithSpecificTime = {
      ...mockUserMessage,
      timestamp: '2024-01-15T14:45:30.000Z'
    };
    
    render(<Message message={messageWithSpecificTime} />);
    
    expect(screen.getByText('14:45')).toBeInTheDocument();
  });

  test('handles long messages with proper wrapping', () => {
    const longMessage = {
      ...mockUserMessage,
      text: 'This is a very long message that should wrap properly in the UI. '.repeat(10)
    };
    
    render(<Message message={longMessage} />);
    
    const messageElement = screen.getByText(longMessage.text);
    expect(messageElement).toHaveClass('whitespace-pre-wrap');
  });
}); 