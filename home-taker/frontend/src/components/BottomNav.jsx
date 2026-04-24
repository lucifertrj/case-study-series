import React from 'react';
import { colors } from '../styles/colors';
import { Icons } from './Icons';

const BottomNav = ({ active, onNavigate }) => (
  <div style={{
    position: 'fixed',
    bottom: 0,
    left: 0,
    right: 0,
    background: colors.warmWhite,
    borderTop: `1px solid ${colors.terracottaLight}40`,
    padding: '8px 0 max(16px, env(safe-area-inset-bottom))',
    display: 'flex',
    justifyContent: 'space-around',
    zIndex: 100,
  }}>
    {[
      { icon: Icons.Home, label: 'Home', id: 'dashboard' },
      { icon: Icons.Calendar, label: 'Schedule', id: 'schedule' },
      { icon: Icons.FileText, label: 'Log', id: 'log' },
      { icon: Icons.MessageCircle, label: 'Chat', id: 'chat' },
    ].map(item => (
      <button
        key={item.id}
        onClick={() => onNavigate(item.id)}
        style={{
          background: 'none',
          border: 'none',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '4px',
          color: active === item.id ? colors.sage : colors.softBrown,
          cursor: 'pointer',
          padding: '8px 16px',
          minWidth: '60px',
        }}
      >
        <item.icon />
        <span style={{ fontSize: '11px', fontFamily: 'system-ui' }}>{item.label}</span>
      </button>
    ))}
  </div>
);

export default BottomNav;
