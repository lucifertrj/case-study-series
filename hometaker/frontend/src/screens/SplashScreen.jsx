import React from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';

const SplashScreen = ({ onStart }) => (
  <div style={{
    minHeight: '100vh',
    // background: `linear-gradient(135deg, ${colors.cream} 0%, ${colors.terracottaLight} 100%)`,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px 24px',
    fontFamily: "'Georgia', serif",
  }}>
    <div style={{
      width: '100px',
      height: '100px',
      borderRadius: '50%',
      background: colors.warmWhite,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      boxShadow: '0 8px 32px rgba(107, 91, 79, 0.15)',
      marginBottom: '28px',
    }}>
      <div style={{ color: colors.terracotta, transform: 'scale(2)' }}>
        <Icons.Heart />
      </div>
    </div>
    <h1 style={{
      fontSize: 'clamp(32px, 8vw, 42px)',
      fontWeight: '400',
      color: colors.charcoal,
      margin: '0 0 8px',
      letterSpacing: '2px'
    }}>
      HOME-TAKER
    </h1>
    <p style={{
      fontSize: '16px',
      color: colors.softBrown,
      margin: '0 0 40px',
      fontStyle: 'italic'
    }}>
      Care, remembered.
    </p>
    <button onClick={onStart} style={{
      background: colors.sage,
      color: colors.warmWhite,
      border: 'none',
      padding: '16px 40px',
      borderRadius: '40px',
      fontSize: '17px',
      fontFamily: "'Georgia', serif",
      cursor: 'pointer',
      boxShadow: '0 4px 20px rgba(139, 154, 125, 0.4)',
    }}>
      Get Started
    </button>
  </div>
);

export default SplashScreen;
