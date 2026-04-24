import React, { useState, useRef, useEffect } from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';
import BottomNav from '../components/BottomNav';

const API_BASE_URL = 'http://localhost:8000';

const Chat = ({ userData, userType, onNavigate }) => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const isCaregiver = userType === 'caregiver';
  const name = userData?.name?.split(' ')[0] || 'there';
  const careRecipientName = userData?.currentRecipient?.name?.split(' ')[0];

  // Initial welcome message
  useEffect(() => {
    if (messages.length === 0) {
      const welcomeMsg = isCaregiver
        ? `Hi ${name}! I'm here to help you care for ${careRecipientName || 'your loved one'}. Ask me anything about their medications, appointments, or health information.`
        : `Hi ${name}! I'm your care assistant. Ask me about your medications, appointments, or anything I can help you remember.`;

      setMessages([{ from: 'ai', text: welcomeMsg }]);
    }
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const suggestions = isCaregiver
    ? [
        `What medications does ${careRecipientName || 'my loved one'} take?`,
        "What allergies should I know about?",
        "Who is the emergency contact?"
      ]
    : [
        "What are my medications?",
        "Who is my emergency contact?",
        "What allergies do I have?"
      ];

  const handleSend = async () => {
    if (!message.trim() || loading) return;

    const userMessage = message.trim();
    setMessage('');
    setMessages(prev => [...prev, { from: 'user', text: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, { from: 'ai', text: data.response }]);
      } else {
        setMessages(prev => [...prev, {
          from: 'ai',
          text: "I'm sorry, I couldn't process that request. Please try again."
        }]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        from: 'ai',
        text: "I'm having trouble connecting to the server. Please make sure the backend is running."
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{
      height: '100vh',
      maxHeight: '100vh',
      background: colors.cream,
      fontFamily: "'Georgia', serif",
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: `1px solid ${colors.terracottaLight}40`,
        background: colors.warmWhite,
        flexShrink: 0,
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          {/* Back Button */}
          <button
            onClick={() => onNavigate('dashboard')}
            style={{
              background: 'none',
              border: 'none',
              padding: '8px',
              margin: '-8px',
              marginRight: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: colors.sage,
              cursor: 'pointer',
            }}
          >
            <Icons.ArrowLeft />
          </button>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${colors.sage}, ${colors.sageDark})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: colors.warmWhite,
            flexShrink: 0,
          }}>
            <Icons.Heart />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1 style={{
              margin: 0,
              fontSize: '17px',
              fontWeight: '500',
              color: colors.charcoal
            }}>
              HOME-TAKER
            </h1>
            <p style={{
              margin: 0,
              fontSize: '12px',
              color: colors.softBrown
            }}>
              Your care assistant
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        padding: '16px',
        paddingBottom: '8px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            display: 'flex',
            justifyContent: msg.from === 'user' ? 'flex-end' : 'flex-start',
            marginBottom: '14px'
          }}>
            <div style={{
              maxWidth: '85%',
              padding: '12px 16px',
              borderRadius: msg.from === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
              background: msg.from === 'user' ? colors.sage : colors.warmWhite,
              color: msg.from === 'user' ? colors.warmWhite : colors.charcoal,
              fontSize: '15px',
              lineHeight: '1.5',
              fontFamily: 'system-ui',
              boxShadow: msg.from === 'ai' ? '0 2px 8px rgba(107, 91, 79, 0.08)' : 'none',
              whiteSpace: 'pre-line',
            }}>
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start',
            marginBottom: '14px'
          }}>
            <div style={{
              padding: '12px 16px',
              borderRadius: '18px 18px 18px 4px',
              background: colors.warmWhite,
              boxShadow: '0 2px 8px rgba(107, 91, 79, 0.08)',
            }}>
              <div style={{
                display: 'flex',
                gap: '4px',
                alignItems: 'center'
              }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: colors.softBrown,
                  animation: 'pulse 1s infinite',
                }} />
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: colors.softBrown,
                  animation: 'pulse 1s infinite 0.2s',
                }} />
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: colors.softBrown,
                  animation: 'pulse 1s infinite 0.4s',
                }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />

        {/* Suggestions - only show if no conversation yet */}
        {messages.length <= 1 && (
          <div style={{ marginTop: '20px' }}>
            <p style={{
              fontSize: '12px',
              color: colors.softBrown,
              marginBottom: '10px'
            }}>
              Try asking:
            </p>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px'
            }}>
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setMessage(s)}
                  style={{
                    background: colors.warmWhite,
                    border: `1px solid ${colors.terracottaLight}`,
                    borderRadius: '12px',
                    padding: '12px 14px',
                    fontSize: '14px',
                    color: colors.charcoal,
                    cursor: 'pointer',
                    fontFamily: 'system-ui',
                    textAlign: 'left',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input - positioned above BottomNav */}
      <div style={{
        background: colors.warmWhite,
        padding: '12px 16px',
        paddingBottom: 'max(12px, env(safe-area-inset-bottom))',
        marginBottom: '70px', // Space for BottomNav
        borderTop: `1px solid ${colors.terracottaLight}40`,
        flexShrink: 0,
      }}>
        <div style={{
          display: 'flex',
          gap: '10px',
          alignItems: 'center',
          background: colors.cream,
          borderRadius: '24px',
          padding: '6px 6px 6px 18px',
          maxWidth: '100%',
        }}>
          <input
            type="text"
            placeholder="Ask me anything..."
            value={message}
            onChange={e => setMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={loading}
            style={{
              flex: 1,
              border: 'none',
              background: 'transparent',
              fontSize: '16px',
              fontFamily: 'system-ui',
              color: colors.charcoal,
              outline: 'none',
              minWidth: 0,
            }}
          />
          <button
            onClick={handleSend}
            disabled={!message.trim() || loading}
            style={{
              background: message.trim() && !loading ? colors.sage : colors.terracottaLight,
              border: 'none',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              flexShrink: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: colors.warmWhite,
              cursor: message.trim() && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            <Icons.Send />
          </button>
        </div>
      </div>

      {/* Bottom Navigation */}
      <BottomNav active="chat" onNavigate={onNavigate} />

      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
          }
        `}
      </style>
    </div>
  );
};

export default Chat;
