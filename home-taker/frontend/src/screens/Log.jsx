import React, { useState } from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';
import { TextInput } from '../components/FormInputs';
import BottomNav from '../components/BottomNav';
import SavingOverlay from '../components/SavingOverlay';

const API_BASE_URL = 'http://localhost:8000';

const Log = ({ userData, userType, onNavigate }) => {
  const [showAddModal, setShowAddModal] = useState(false);
  const [logType, setLogType] = useState(null);
  const [newEntry, setNewEntry] = useState({ title: '', details: '' });
  const [logs, setLogs] = useState([]);
  const [saving, setSaving] = useState(false);

  const logTypes = [
    { id: 'medication', icon: Icons.Pill, label: 'Medication Change', color: colors.terracotta },
    { id: 'contact', icon: Icons.Phone, label: 'New Contact', color: colors.sage },
    { id: 'appointment', icon: Icons.Calendar, label: 'Appointment Note', color: colors.dustyRose },
    { id: 'memory', icon: Icons.Star, label: 'Remember This', color: colors.gold },
    { id: 'photo', icon: Icons.Camera, label: 'Photo Log', color: colors.softBrown },
    { id: 'voice', icon: Icons.Mic, label: 'Voice Note', color: colors.warmBrown },
  ];

  const handleSaveLog = async () => {
    if (!newEntry.title.trim()) return;

    setSaving(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: logType,
          title: newEntry.title,
          details: newEntry.details,
        }),
      });

      if (response.ok) {
        // Add to local logs
        const newLog = {
          type: logType,
          title: newEntry.title,
          desc: newEntry.details,
          time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
          isNew: true,
        };

        setLogs(prev => [{
          date: 'Today',
          items: [newLog, ...(prev[0]?.date === 'Today' ? prev[0].items : [])],
        }, ...prev.filter(g => g.date !== 'Today')]);

        // Reset form
        setNewEntry({ title: '', details: '' });
        setLogType(null);
        setShowAddModal(false);
      }
    } catch (error) {
      console.error('Failed to save log:', error);
    } finally {
      setSaving(false);
    }
  };

  // Show empty state message
  const emptyState = logs.length === 0 || (logs.length === 1 && logs[0].items.length === 0);

  return (
    <div style={{
      minHeight: '100vh',
      background: colors.cream,
      fontFamily: "'Georgia', serif",
      paddingBottom: '90px'
    }}>
      <div style={{ padding: '20px' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '20px'
        }}>
          <h1 style={{
            fontSize: 'clamp(22px, 5vw, 26px)',
            fontWeight: '400',
            color: colors.charcoal,
            margin: 0
          }}>
            Activity Log
          </h1>
          <button
            onClick={() => setShowAddModal(true)}
            style={{
              background: colors.sage,
              border: 'none',
              borderRadius: '12px',
              padding: '10px 16px',
              color: colors.warmWhite,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '14px',
            }}
          >
            <Icons.Plus /> Add New
          </button>
        </div>

        {emptyState ? (
          <div style={{
            background: colors.warmWhite,
            borderRadius: '18px',
            padding: '40px 20px',
            textAlign: 'center',
            boxShadow: '0 4px 20px rgba(107, 91, 79, 0.08)',
          }}>
            <div style={{
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              background: `${colors.sage}20`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              color: colors.sage,
            }}>
              <Icons.FileText />
            </div>
            <h3 style={{
              margin: '0 0 8px',
              fontSize: '18px',
              color: colors.charcoal,
              fontWeight: '500'
            }}>
              No logs yet
            </h3>
            <p style={{
              margin: '0 0 20px',
              fontSize: '14px',
              color: colors.softBrown,
              lineHeight: '1.5'
            }}>
              Start tracking medications, appointments, contacts, and important memories.
            </p>
            <button
              onClick={() => setShowAddModal(true)}
              style={{
                background: colors.sage,
                border: 'none',
                borderRadius: '10px',
                padding: '12px 24px',
                color: colors.warmWhite,
                fontSize: '15px',
                cursor: 'pointer',
              }}
            >
              Add your first log
            </button>
          </div>
        ) : (
          logs.map((group, gi) => (
            <div key={gi} style={{ marginBottom: '24px' }}>
              <h3 style={{
                margin: '0 0 12px',
                fontSize: '13px',
                color: colors.softBrown,
                fontWeight: '500',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                {group.date}
              </h3>
              {group.items.map((item, i) => (
                <div key={i} style={{
                  background: colors.warmWhite,
                  borderRadius: '14px',
                  padding: '14px 16px',
                  marginBottom: '10px',
                  boxShadow: '0 2px 10px rgba(107, 91, 79, 0.06)',
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '12px'
                  }}>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        marginBottom: '4px'
                      }}>
                        <p style={{
                          margin: 0,
                          fontSize: '15px',
                          color: colors.charcoal,
                          fontWeight: '500',
                          fontFamily: 'system-ui'
                        }}>
                          {item.title}
                        </p>
                        {item.isNew && (
                          <span style={{
                            background: `${colors.terracotta}20`,
                            color: colors.terracotta,
                            fontSize: '10px',
                            padding: '2px 8px',
                            borderRadius: '10px'
                          }}>
                            New
                          </span>
                        )}
                      </div>
                      {item.desc && (
                        <p style={{
                          margin: '0 0 6px',
                          fontSize: '14px',
                          color: colors.softBrown,
                          lineHeight: '1.4'
                        }}>
                          {item.desc}
                        </p>
                      )}
                      <p style={{
                        margin: 0,
                        fontSize: '11px',
                        color: `${colors.softBrown}99`
                      }}>
                        {item.time}
                      </p>
                    </div>
                    <button style={{
                      background: 'none',
                      border: 'none',
                      color: colors.softBrown,
                      cursor: 'pointer',
                      padding: '4px'
                    }}>
                      <Icons.Edit />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Add Modal */}
      {showAddModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'center',
            zIndex: 200,
            padding: '20px',
          }}
          onClick={() => { setShowAddModal(false); setLogType(null); }}
        >
          <div
            style={{
              background: colors.cream,
              borderRadius: '24px 24px 0 0',
              padding: '24px',
              width: '100%',
              maxWidth: '500px',
              maxHeight: '80vh',
              overflow: 'auto',
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px'
            }}>
              <h2 style={{
                margin: 0,
                fontSize: '20px',
                color: colors.charcoal
              }}>
                {logType ? 'Add Entry' : 'What to log?'}
              </h2>
              <button
                onClick={() => { setShowAddModal(false); setLogType(null); }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: colors.softBrown,
                  cursor: 'pointer'
                }}
              >
                <Icons.X />
              </button>
            </div>

            {!logType ? (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '12px'
              }}>
                {logTypes.map(lt => (
                  <button
                    key={lt.id}
                    onClick={() => setLogType(lt.id)}
                    style={{
                      background: colors.warmWhite,
                      border: 'none',
                      borderRadius: '16px',
                      padding: '20px 16px',
                      cursor: 'pointer',
                      textAlign: 'center',
                    }}
                  >
                    <div style={{
                      width: '48px',
                      height: '48px',
                      borderRadius: '50%',
                      background: `${lt.color}20`,
                      margin: '0 auto 10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: lt.color,
                    }}>
                      <lt.icon />
                    </div>
                    <p style={{
                      margin: 0,
                      fontSize: '14px',
                      color: colors.charcoal
                    }}>
                      {lt.label}
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <div>
                <button
                  onClick={() => setLogType(null)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: colors.sage,
                    fontSize: '14px',
                    cursor: 'pointer',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <Icons.ChevronLeft /> Back
                </button>
                <TextInput
                  label="Title"
                  placeholder="What happened?"
                  value={newEntry.title}
                  onChange={v => setNewEntry(p => ({ ...p, title: v }))}
                  required
                />
                <TextInput
                  label="Details"
                  placeholder="Add more context (optional)"
                  value={newEntry.details}
                  onChange={v => setNewEntry(p => ({ ...p, details: v }))}
                />
                <button
                  onClick={handleSaveLog}
                  disabled={!newEntry.title.trim() || saving}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '12px',
                    border: 'none',
                    background: newEntry.title.trim() ? colors.sage : colors.terracottaLight,
                    color: colors.warmWhite,
                    fontSize: '16px',
                    cursor: newEntry.title.trim() ? 'pointer' : 'not-allowed',
                    marginTop: '8px',
                  }}
                >
                  {saving ? 'Saving...' : 'Save to Memory'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <BottomNav active="log" onNavigate={onNavigate} />

      {/* Saving overlay with animation */}
      {saving && <SavingOverlay message="Saving to memory..." />}
    </div>
  );
};

export default Log;
