import React, { useState, useEffect } from 'react';
import { colors } from '../styles/colors';

const savingQuotes = [
  "Every moment remembered is a treasure kept safe...",
  "Building your story, one memory at a time...",
  "What we keep in heart, we never truly lose...",
  "Memories are the diary we carry within us...",
  "Creating a bridge between yesterday and tomorrow...",
];

const SavingOverlay = ({ message = "Saving your preferences..." }) => {
  const [quoteIndex, setQuoteIndex] = useState(0);
  const [fadeState, setFadeState] = useState('in');

  useEffect(() => {
    const interval = setInterval(() => {
      // Start fade out
      setFadeState('out');

      // After fade out, change quote and fade in
      setTimeout(() => {
        setQuoteIndex(prev => (prev + 1) % savingQuotes.length);
        setFadeState('in');
      }, 500);
    }, 6000); // Change quote every 6 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(61, 56, 50, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
    }}>
      <div style={{
        background: colors.warmWhite,
        padding: '32px 40px',
        borderRadius: '24px',
        textAlign: 'center',
        maxWidth: '320px',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
      }}>
        {/* Animated spinner */}
        <div style={{
          width: '60px',
          height: '60px',
          margin: '0 auto 20px',
          position: 'relative',
        }}>
          {/* Outer pulsing ring */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            border: `3px solid ${colors.sage}30`,
            animation: 'pulse 2s ease-in-out infinite',
          }} />

          {/* Spinning arc */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            border: `3px solid transparent`,
            borderTopColor: colors.sage,
            borderRightColor: colors.sage,
            animation: 'spin 1.2s linear infinite',
          }} />

          {/* Center icon */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: colors.sage,
            animation: 'heartbeat 1.5s ease-in-out infinite',
          }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
            </svg>
          </div>
        </div>

        {/* Main message */}
        <p style={{
          margin: '0 0 16px',
          fontSize: '17px',
          fontWeight: '500',
          color: colors.charcoal,
          fontFamily: "'Georgia', serif",
        }}>
          {message}
        </p>

        {/* Rotating quote */}
        <p style={{
          margin: 0,
          fontSize: '14px',
          color: colors.softBrown,
          fontStyle: 'italic',
          fontFamily: "'Georgia', serif",
          lineHeight: '1.5',
          minHeight: '42px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          opacity: fadeState === 'in' ? 1 : 0,
          transition: 'opacity 0.5s ease-in-out',
        }}>
          "{savingQuotes[quoteIndex]}"
        </p>

        {/* Progress dots */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '8px',
          marginTop: '20px',
        }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: colors.sage,
                opacity: 0.4,
                animation: `bounce 1.4s ease-in-out ${i * 0.2}s infinite`,
              }}
            />
          ))}
        </div>
      </div>

      {/* CSS Animations */}
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }

          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.2; }
          }

          @keyframes heartbeat {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.1); }
          }

          @keyframes bounce {
            0%, 80%, 100% {
              transform: translateY(0);
              opacity: 0.4;
            }
            40% {
              transform: translateY(-6px);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
};

export default SavingOverlay;
