import React, { useState } from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';
import BottomNav from '../components/BottomNav';

const Schedule = ({ userData, userType, onNavigate }) => {
  const [selectedDate, setSelectedDate] = useState(15);
  const days = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
  const isCaregiver = userType === 'caregiver';

  const events = [
    { time: '8:00 AM', title: 'Morning medications', type: 'medication', color: colors.terracotta },
    { time: '10:00 AM', title: isCaregiver ? "Mom's physical therapy" : 'Walk in the garden', type: 'activity', color: colors.sage },
    { time: '2:00 PM', title: isCaregiver ? "Dad's eye doctor" : 'Dr. Smith checkup', type: 'appointment', color: colors.dustyRose },
    { time: '4:00 PM', title: 'Water reminder', type: 'reminder', color: colors.gold },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: colors.cream,
      fontFamily: "'Georgia', serif",
      paddingBottom: '90px'
    }}>
      <div style={{ padding: '20px' }}>
        <h1 style={{
          fontSize: 'clamp(22px, 5vw, 26px)',
          fontWeight: '400',
          color: colors.charcoal,
          margin: '0 0 20px'
        }}>
          Schedule
        </h1>

        {/* Calendar */}
        <div style={{
          background: colors.warmWhite,
          borderRadius: '18px',
          padding: '18px',
          marginBottom: '20px',
          boxShadow: '0 4px 20px rgba(107, 91, 79, 0.08)'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '16px'
          }}>
            <button style={{
              background: 'none',
              border: 'none',
              color: colors.softBrown,
              cursor: 'pointer'
            }}>
              <Icons.ChevronLeft />
            </button>
            <h3 style={{
              margin: 0,
              fontSize: '17px',
              color: colors.charcoal
            }}>
              January 2026
            </h3>
            <button style={{
              background: 'none',
              border: 'none',
              color: colors.softBrown,
              cursor: 'pointer'
            }}>
              <Icons.ChevronRight />
            </button>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(7, 1fr)',
            gap: '4px',
            textAlign: 'center'
          }}>
            {days.map((d, i) => (
              <div key={`day-${i}`} style={{
                fontSize: '12px',
                color: colors.softBrown,
                padding: '8px 0'
              }}>
                {d}
              </div>
            ))}
            {[...Array(31)].map((_, i) => (
              <button
                key={i}
                onClick={() => setSelectedDate(i + 1)}
                style={{
                  padding: '10px 0',
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontFamily: 'system-ui',
                  background: i + 1 === selectedDate ? colors.sage : 'transparent',
                  color: i + 1 === selectedDate ? colors.warmWhite : colors.charcoal,
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Events */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '12px'
        }}>
          <h3 style={{
            margin: 0,
            fontSize: '15px',
            color: colors.softBrown,
            fontWeight: '500'
          }}>
            January {selectedDate}
          </h3>
          <button style={{
            background: colors.sage,
            border: 'none',
            borderRadius: '10px',
            padding: '8px 14px',
            color: colors.warmWhite,
            fontSize: '13px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            <Icons.Plus /> Add
          </button>
        </div>

        {events.map((event, i) => (
          <div key={i} style={{
            background: colors.warmWhite,
            borderRadius: '14px',
            padding: '14px 16px',
            marginBottom: '10px',
            borderLeft: `4px solid ${event.color}`,
            boxShadow: '0 2px 10px rgba(107, 91, 79, 0.06)',
          }}>
            <p style={{
              margin: '0 0 4px',
              fontSize: '15px',
              color: colors.charcoal,
              fontFamily: 'system-ui'
            }}>
              {event.title}
            </p>
            <p style={{
              margin: 0,
              fontSize: '13px',
              color: colors.softBrown
            }}>
              {event.time}
            </p>
          </div>
        ))}
      </div>

      <BottomNav active="schedule" onNavigate={onNavigate} />
    </div>
  );
};

export default Schedule;
