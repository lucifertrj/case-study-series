import React from 'react';
import { colors } from '../styles/colors';
import { Icons } from './Icons';

export const TextInput = ({ label, placeholder, value, onChange, type = 'text', helper, required = false, error = false }) => (
  <div style={{ marginBottom: '16px' }}>
    {label && (
      <label style={{
        display: 'block',
        fontSize: '14px',
        color: error ? colors.terracotta : colors.softBrown,
        marginBottom: '6px',
        fontFamily: 'system-ui'
      }}>
        {label}
        {required && <span style={{ color: colors.terracotta, marginLeft: '4px' }}>*</span>}
      </label>
    )}
    <input
      type={type}
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        width: '100%',
        background: colors.warmWhite,
        border: `2px solid ${error ? colors.terracotta : colors.terracottaLight}`,
        borderRadius: '12px',
        padding: '14px 16px',
        fontSize: '16px',
        fontFamily: 'system-ui',
        color: colors.charcoal,
        outline: 'none',
        boxSizing: 'border-box',
        transition: 'border-color 0.2s',
      }}
      onFocus={(e) => e.target.style.borderColor = error ? colors.terracotta : colors.sage}
      onBlur={(e) => e.target.style.borderColor = error ? colors.terracotta : colors.terracottaLight}
    />
    {error && (
      <p style={{
        margin: '6px 0 0',
        fontSize: '12px',
        color: colors.terracotta,
      }}>
        This field is required
      </p>
    )}
    {helper && !error && (
      <p style={{
        margin: '6px 0 0',
        fontSize: '12px',
        color: colors.softBrown,
        fontStyle: 'italic'
      }}>
        {helper}
      </p>
    )}
  </div>
);

export const YesNoInput = ({ value, onChange }) => (
  <div style={{ display: 'flex', gap: '12px' }}>
    {['Yes', 'No'].map(opt => (
      <button
        key={opt}
        onClick={() => onChange(opt)}
        style={{
          flex: 1,
          padding: '16px',
          borderRadius: '12px',
          border: `2px solid ${value === opt ? colors.sage : colors.terracottaLight}`,
          background: value === opt ? `${colors.sage}15` : colors.warmWhite,
          fontSize: '16px',
          fontFamily: "'Georgia', serif",
          color: colors.charcoal,
          cursor: 'pointer',
        }}
      >
        {opt}
      </button>
    ))}
  </div>
);

export const MultiSelectInput = ({ options, value = [], onChange }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
    {options.map(opt => {
      const selected = value.includes(opt);
      return (
        <button
          key={opt}
          onClick={() => onChange(selected ? value.filter(x => x !== opt) : [...value, opt])}
          style={{
            padding: '12px 16px',
            borderRadius: '10px',
            border: `2px solid ${selected ? colors.sage : colors.terracottaLight}`,
            background: selected ? `${colors.sage}15` : colors.warmWhite,
            fontSize: '15px',
            fontFamily: 'system-ui',
            color: colors.charcoal,
            cursor: 'pointer',
            textAlign: 'left',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          <div style={{
            width: '20px',
            height: '20px',
            borderRadius: '5px',
            border: `2px solid ${selected ? colors.sage : colors.softBrown}`,
            background: selected ? colors.sage : 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: colors.warmWhite,
            flexShrink: 0,
          }}>
            {selected && <Icons.Check />}
          </div>
          {opt}
        </button>
      );
    })}
  </div>
);

export const TimeSelect = ({ value, onChange, label, required = false }) => {
  const times = [
    '6:00 AM', '7:00 AM', '8:00 AM', '9:00 AM', '10:00 AM', '11:00 AM',
    '12:00 PM', '1:00 PM', '2:00 PM', '3:00 PM', '4:00 PM', '5:00 PM',
    '6:00 PM', '7:00 PM', '8:00 PM', '9:00 PM', '10:00 PM'
  ];

  return (
    <div style={{ marginBottom: '16px' }}>
      {label && (
        <label style={{
          display: 'block',
          fontSize: '14px',
          color: colors.softBrown,
          marginBottom: '6px',
          fontFamily: 'system-ui'
        }}>
          {label}
          {required && <span style={{ color: colors.terracotta, marginLeft: '4px' }}>*</span>}
        </label>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          background: colors.warmWhite,
          border: `2px solid ${colors.terracottaLight}`,
          borderRadius: '12px',
          padding: '14px 16px',
          fontSize: '16px',
          fontFamily: 'system-ui',
          color: colors.charcoal,
          outline: 'none',
        }}
      >
        <option value="">Select time</option>
        {times.map(t => <option key={t} value={t}>{t}</option>)}
      </select>
    </div>
  );
};
