import React from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';

const UserTypeScreen = ({ onSelect }) => (
  <div style={{
    minHeight: '100vh',
    background: colors.cream,
    padding: '48px 20px',
    fontFamily: "'Georgia', serif",
  }}>
    <h2 style={{
      fontSize: 'clamp(24px, 6vw, 28px)',
      fontWeight: '400',
      color: colors.charcoal,
      margin: '0 0 8px',
      textAlign: 'center'
    }}>
      Welcome to HOME-TAKER
    </h2>
    <p style={{
      fontSize: '15px',
      color: colors.softBrown,
      textAlign: 'center',
      margin: '0 0 40px'
    }}>
      How will you be using the app?
    </p>

    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      maxWidth: '400px',
      margin: '0 auto'
    }}>
      {[
        {
          type: 'elderly',
          icon: Icons.User,
          title: 'For Myself',
          desc: 'I want to manage my own care & daily routines',
          color: colors.terracotta
        },
        {
          type: 'caregiver',
          icon: Icons.Users,
          title: 'Caring for Family',
          desc: 'I help manage care for my parent(s) or loved ones',
          color: colors.sage
        },
      ].map(item => (
        <button key={item.type} onClick={() => onSelect(item.type)} style={{
          background: colors.warmWhite,
          border: `2px solid ${colors.terracottaLight}`,
          borderRadius: '16px',
          padding: '24px 20px',
          cursor: 'pointer',
          textAlign: 'left',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div style={{
              width: '52px',
              height: '52px',
              borderRadius: '50%',
              background: `${item.color}20`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: item.color,
              flexShrink: 0,
            }}>
              <item.icon />
            </div>
            <div>
              <h3 style={{
                margin: '0 0 4px',
                fontSize: '18px',
                color: colors.charcoal,
                fontWeight: '500'
              }}>
                {item.title}
              </h3>
              <p style={{
                margin: 0,
                fontSize: '13px',
                color: colors.softBrown,
                fontFamily: 'system-ui'
              }}>
                {item.desc}
              </p>
            </div>
          </div>
        </button>
      ))}
    </div>
  </div>
);

export default UserTypeScreen;
