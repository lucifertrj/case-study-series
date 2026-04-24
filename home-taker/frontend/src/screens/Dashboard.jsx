import React from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';
import BottomNav from '../components/BottomNav';

const Dashboard = ({ userData, userType, onNavigate }) => {
  const name = userData?.name?.split(' ')[0] || 'Friend';
  const isCaregiver = userType === 'caregiver';
  const careRecipientName = userData?.currentRecipient?.name?.split(' ')[0];

  // Get time of day for greeting
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  // Build today's tasks from user data
  const getTodayTasks = () => {
    const tasks = [];

    // Add medication tasks if user has medications
    if (userData?.medications?.length > 0) {
      userData.medications.forEach(med => {
        if (med.time) {
          tasks.push({
            icon: Icons.Pill,
            title: `${med.name} ${med.dosage || ''}`.trim(),
            time: med.time,
            done: false,
            type: 'medication'
          });
        }
      });
    }

    // Add water reminder if enabled
    if (userData?.waterReminder === 'Yes') {
      tasks.push({
        icon: Icons.Droplet,
        title: 'Drink water',
        time: userData.waterFrequency || 'Throughout the day',
        done: false,
        type: 'reminder'
      });
    }

    // If no tasks, show a welcome message
    if (tasks.length === 0) {
      tasks.push({
        icon: Icons.Calendar,
        title: 'No tasks scheduled yet',
        time: 'Add medications or reminders',
        done: false,
        type: 'info'
      });
    }

    return tasks;
  };

  const todayTasks = getTodayTasks();

  return (
    <div style={{
      minHeight: '100vh',
      background: colors.cream,
      fontFamily: "'Georgia', serif",
      paddingBottom: '90px'
    }}>
      {/* Header */}
      <div style={{
        background: `linear-gradient(135deg, ${colors.sage} 0%, ${colors.sageDark} 100%)`,
        padding: '24px 20px 40px',
        borderRadius: '0 0 28px 28px',
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '20px'
        }}>
          <div>
            <p style={{
              margin: 0,
              fontSize: '13px',
              color: `${colors.warmWhite}99`
            }}>
              {getGreeting()}
            </p>
            <h1 style={{
              margin: '4px 0 0',
              fontSize: 'clamp(24px, 6vw, 28px)',
              fontWeight: '400',
              color: colors.warmWhite
            }}>
              {name}
            </h1>
          </div>
          <button style={{
            background: `${colors.warmWhite}20`,
            border: 'none',
            borderRadius: '50%',
            width: '44px',
            height: '44px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: colors.warmWhite,
            cursor: 'pointer',
          }}>
            <Icons.Settings />
          </button>
        </div>

        {/* Quick Actions */}
        <div style={{ display: 'flex', gap: '10px' }}>
          <button style={{
            flex: 1,
            background: colors.terracotta,
            border: 'none',
            borderRadius: '14px',
            padding: '14px',
            color: colors.warmWhite,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            fontSize: '14px',
            fontFamily: 'system-ui',
            fontWeight: '500',
          }}>
            <Icons.AlertCircle /> Emergency
          </button>
          <button
            onClick={() => onNavigate('log')}
            style={{
              flex: 1,
              background: `${colors.warmWhite}20`,
              border: 'none',
              borderRadius: '14px',
              padding: '14px',
              color: colors.warmWhite,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontSize: '14px',
              fontFamily: 'system-ui',
              fontWeight: '500',
            }}
          >
            <Icons.Plus /> Quick Log
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: '20px', marginTop: '-20px' }}>
        {/* Care Recipient Info (for caregivers) */}
        {isCaregiver && careRecipientName && (
          <div style={{
            background: `${colors.dustyRose}20`,
            borderRadius: '14px',
            padding: '14px 16px',
            marginBottom: '16px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: colors.dustyRose,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: colors.warmWhite,
            }}>
              <Icons.User />
            </div>
            <div>
              <p style={{
                margin: 0,
                fontSize: '15px',
                fontWeight: '500',
                color: colors.charcoal
              }}>
                Caring for {careRecipientName}
              </p>
              <p style={{
                margin: '2px 0 0',
                fontSize: '13px',
                color: colors.softBrown
              }}>
                {userData?.currentRecipient?.relationship}
              </p>
            </div>
          </div>
        )}

        {/* Today's Summary */}
        <div style={{
          background: colors.warmWhite,
          borderRadius: '18px',
          padding: '18px',
          boxShadow: '0 4px 20px rgba(107, 91, 79, 0.08)',
          marginBottom: '16px',
        }}>
          <h3 style={{
            margin: '0 0 14px',
            fontSize: '15px',
            color: colors.softBrown,
            fontWeight: '500'
          }}>
            Today
          </h3>
          {todayTasks.map((item, i) => (
            <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '12px 0',
              borderBottom: i < todayTasks.length - 1 ? `1px solid ${colors.terracottaLight}30` : 'none',
            }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                background: item.done ? `${colors.sage}20` : `${colors.terracotta}15`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: item.done ? colors.sage : colors.terracotta,
              }}>
                <item.icon />
              </div>
              <div style={{ flex: 1 }}>
                <p style={{
                  margin: '0 0 2px',
                  fontSize: '15px',
                  color: colors.charcoal,
                  fontFamily: 'system-ui',
                  textDecoration: item.done ? 'line-through' : 'none',
                  opacity: item.done ? 0.6 : 1
                }}>
                  {item.title}
                </p>
                <p style={{
                  margin: 0,
                  fontSize: '12px',
                  color: colors.softBrown
                }}>
                  {item.time}
                </p>
              </div>
              {item.done && <div style={{ color: colors.sage }}><Icons.Check /></div>}
            </div>
          ))}
        </div>

        {/* Profile Summary */}
        <div style={{
          background: colors.warmWhite,
          borderRadius: '18px',
          padding: '18px',
          boxShadow: '0 4px 20px rgba(107, 91, 79, 0.08)',
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '14px'
          }}>
            <h3 style={{
              margin: 0,
              fontSize: '15px',
              color: colors.softBrown,
              fontWeight: '500'
            }}>
              Quick Info
            </h3>
            <button
              onClick={() => onNavigate('chat')}
              style={{
                background: 'none',
                border: 'none',
                color: colors.sage,
                fontSize: '13px',
                cursor: 'pointer'
              }}
            >
              Ask me anything
            </button>
          </div>

          {/* Show relevant user info */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {userData?.bloodType && (
              <div style={{
                background: `${colors.terracotta}10`,
                borderRadius: '10px',
                padding: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Icons.Droplet style={{ color: colors.terracotta }} />
                <span style={{ fontSize: '14px', color: colors.charcoal }}>
                  Blood Type: <strong>{userData.bloodType}</strong>
                </span>
              </div>
            )}

            {userData?.emergencyContact?.name && (
              <div style={{
                background: `${colors.sage}10`,
                borderRadius: '10px',
                padding: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Icons.Phone style={{ color: colors.sage }} />
                <span style={{ fontSize: '14px', color: colors.charcoal }}>
                  Emergency: <strong>{userData.emergencyContact.name}</strong> ({userData.emergencyContact.phone})
                </span>
              </div>
            )}

            {userData?.doctor?.name && (
              <div style={{
                background: `${colors.dustyRose}10`,
                borderRadius: '10px',
                padding: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Icons.User style={{ color: colors.dustyRose }} />
                <span style={{ fontSize: '14px', color: colors.charcoal }}>
                  Doctor: <strong>{userData.doctor.name}</strong>
                </span>
              </div>
            )}

            {userData?.allergies?.length > 0 && !userData.allergies.includes('No known allergies') && (
              <div style={{
                background: `${colors.gold}15`,
                borderRadius: '10px',
                padding: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <Icons.AlertCircle style={{ color: colors.gold }} />
                <span style={{ fontSize: '14px', color: colors.charcoal }}>
                  Allergies: <strong>{userData.allergies.join(', ')}</strong>
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      <BottomNav active="dashboard" onNavigate={onNavigate} />
    </div>
  );
};

export default Dashboard;
