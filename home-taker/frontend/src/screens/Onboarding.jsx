import React, { useState } from 'react';
import { colors } from '../styles/colors';
import { Icons } from '../components/Icons';
import { TextInput, YesNoInput, MultiSelectInput, TimeSelect } from '../components/FormInputs';

const Onboarding = ({ userType, onComplete }) => {
  const [step, setStep] = useState(0);
  const [errors, setErrors] = useState({});
  const [data, setData] = useState({
    name: '',
    dateOfBirth: '',
    bloodType: '',
    phone: '',
    emergencyContact: { name: '', relationship: '', phone: '' },
    doctor: { name: '', clinic: '', phone: '' },
    allergies: [],
    customAllergy: '',
    conditions: [],
    customCondition: '',
    medications: [],
    diet: '',
    dietNotes: '',
    waterReminder: '',
    waterFrequency: '',
    medicationReminder: '',
    wakeTime: '',
    sleepTime: '',
    // Caregiver specific
    careRecipients: [],
    currentRecipient: { name: '', relationship: '', dateOfBirth: '', phone: '', address: '' },
    challenges: [],
    familyHelpers: [],
  });

  const [currentMed, setCurrentMed] = useState({
    name: '',
    dosage: '',
    frequency: '',
    time: '',
    color: ''
  });

  const update = (field, value) => {
    setData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: false }));
    }
  };

  const updateNested = (parent, field, value) => {
    setData(prev => ({
      ...prev,
      [parent]: { ...prev[parent], [field]: value }
    }));
    // Clear nested error
    const errorKey = `${parent}.${field}`;
    if (errors[errorKey]) {
      setErrors(prev => ({ ...prev, [errorKey]: false }));
    }
  };

  const addMed = () => {
    if (currentMed.name) {
      update('medications', [...data.medications, { ...currentMed, id: Date.now() }]);
      setCurrentMed({ name: '', dosage: '', frequency: '', time: '', color: '' });
    }
  };

  const isCaregiver = userType === 'caregiver';

  // Validation rules per step
  const validateStep = (stepIndex) => {
    const newErrors = {};

    if (isCaregiver) {
      // Caregiver validation
      if (stepIndex === 0) {
        if (!data.name.trim()) newErrors.name = true;
        if (!data.phone.trim()) newErrors.phone = true;
      } else if (stepIndex === 1) {
        if (!data.currentRecipient.name.trim()) newErrors['currentRecipient.name'] = true;
        if (!data.currentRecipient.relationship.trim()) newErrors['currentRecipient.relationship'] = true;
        if (!data.currentRecipient.dateOfBirth.trim()) newErrors['currentRecipient.dateOfBirth'] = true;
      }
    } else {
      // Elderly validation
      if (stepIndex === 0) {
        if (!data.name.trim()) newErrors.name = true;
        if (!data.dateOfBirth.trim()) newErrors.dateOfBirth = true;
      } else if (stepIndex === 1) {
        if (!data.emergencyContact.name.trim()) newErrors['emergencyContact.name'] = true;
        if (!data.emergencyContact.relationship.trim()) newErrors['emergencyContact.relationship'] = true;
        if (!data.emergencyContact.phone.trim()) newErrors['emergencyContact.phone'] = true;
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleContinue = () => {
    if (step < steps.length - 1) {
      // Validate mandatory steps (first 2 steps for both user types)
      if (step <= 1 && !validateStep(step)) {
        return; // Don't advance if validation fails
      }
      setStep(step + 1);
    } else {
      // Final step - complete onboarding
      onComplete(data);
    }
  };

  const canSkip = () => {
    // Can't skip first 2 steps (mandatory info)
    return step > 1;
  };

  const steps = isCaregiver ? [
    // Caregiver Steps
    {
      title: "About You",
      subtitle: "As the primary caregiver",
      required: true,
      content: (
        <>
          <TextInput
            label="Your full name"
            placeholder="Enter your full name"
            value={data.name}
            onChange={v => update('name', v)}
            required
            error={errors.name}
          />
          <TextInput
            label="Your phone number"
            placeholder="Enter your phone number"
            type="tel"
            value={data.phone}
            onChange={v => update('phone', v)}
            required
            error={errors.phone}
            helper="Listed as emergency contact for your family"
          />
        </>
      )
    },
    {
      title: "Who are you caring for?",
      subtitle: "Tell us about your loved one",
      required: true,
      content: (
        <>
          <TextInput
            label="Their name"
            placeholder="Enter their name"
            value={data.currentRecipient.name}
            onChange={v => updateNested('currentRecipient', 'name', v)}
            required
            error={errors['currentRecipient.name']}
          />
          <TextInput
            label="Relationship"
            placeholder="e.g., Mother, Father, Aunt"
            value={data.currentRecipient.relationship}
            onChange={v => updateNested('currentRecipient', 'relationship', v)}
            required
            error={errors['currentRecipient.relationship']}
          />
          <TextInput
            label="Date of birth"
            placeholder="Enter their date of birth"
            value={data.currentRecipient.dateOfBirth}
            onChange={v => updateNested('currentRecipient', 'dateOfBirth', v)}
            required
            error={errors['currentRecipient.dateOfBirth']}
          />
          <TextInput
            label="Their phone"
            placeholder="Enter their phone number"
            type="tel"
            value={data.currentRecipient.phone}
            onChange={v => updateNested('currentRecipient', 'phone', v)}
          />
          <TextInput
            label="Address"
            placeholder="Enter their address"
            value={data.currentRecipient.address}
            onChange={v => updateNested('currentRecipient', 'address', v)}
          />
        </>
      )
    },
    {
      title: "Their Doctor",
      subtitle: "Primary care physician (optional)",
      required: false,
      content: (
        <>
          <TextInput
            label="Doctor's name"
            placeholder="Enter doctor's name"
            value={data.doctor.name}
            onChange={v => updateNested('doctor', 'name', v)}
          />
          <TextInput
            label="Clinic/Hospital"
            placeholder="Enter clinic name"
            value={data.doctor.clinic}
            onChange={v => updateNested('doctor', 'clinic', v)}
          />
          <TextInput
            label="Phone"
            placeholder="Enter clinic phone"
            type="tel"
            value={data.doctor.phone}
            onChange={v => updateNested('doctor', 'phone', v)}
          />
          <div style={{ marginTop: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '14px',
              color: colors.softBrown,
              marginBottom: '8px'
            }}>
              Blood type
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown'].map(bt => (
                <button
                  key={bt}
                  onClick={() => update('bloodType', bt)}
                  style={{
                    padding: '8px 14px',
                    borderRadius: '8px',
                    border: `2px solid ${data.bloodType === bt ? colors.sage : colors.terracottaLight}`,
                    background: data.bloodType === bt ? `${colors.sage}15` : colors.warmWhite,
                    fontSize: '14px',
                    cursor: 'pointer',
                    color: colors.charcoal,
                  }}
                >
                  {bt}
                </button>
              ))}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Allergies",
      subtitle: "Select all that apply (optional)",
      required: false,
      content: (
        <>
          <MultiSelectInput
            options={['Penicillin', 'Sulfa drugs', 'Aspirin', 'Ibuprofen', 'Latex', 'Shellfish', 'Peanuts', 'No known allergies']}
            value={data.allergies}
            onChange={v => update('allergies', v)}
          />
          <div style={{ marginTop: '16px' }}>
            <TextInput
              label="Other allergies"
              placeholder="Enter any other allergies"
              value={data.customAllergy}
              onChange={v => update('customAllergy', v)}
            />
          </div>
        </>
      )
    },
    {
      title: "Medical Conditions",
      subtitle: "Current health conditions (optional)",
      required: false,
      content: (
        <MultiSelectInput
          options={['High Blood Pressure', 'Type 2 Diabetes', 'Heart Disease', 'Arthritis', 'COPD', "Dementia/Alzheimer's", 'Kidney Disease', 'Depression', 'None']}
          value={data.conditions}
          onChange={v => update('conditions', v)}
        />
      )
    },
    {
      title: "Medications",
      subtitle: "What medications do they take? (optional)",
      required: false,
      content: (
        <>
          {data.medications.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              {data.medications.map(med => (
                <div key={med.id} style={{
                  background: `${colors.sage}10`,
                  borderRadius: '10px',
                  padding: '12px',
                  marginBottom: '8px',
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <div>
                    <p style={{
                      margin: '0 0 2px',
                      fontSize: '15px',
                      fontWeight: '500',
                      color: colors.charcoal
                    }}>
                      {med.name} {med.dosage}
                    </p>
                    <p style={{
                      margin: 0,
                      fontSize: '13px',
                      color: colors.softBrown
                    }}>
                      {med.frequency} {med.time && `at ${med.time}`} {med.color && `• ${med.color}`}
                    </p>
                  </div>
                  <button
                    onClick={() => update('medications', data.medications.filter(m => m.id !== med.id))}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: colors.terracotta,
                      cursor: 'pointer'
                    }}
                  >
                    <Icons.Trash />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div style={{
            background: colors.warmWhite,
            borderRadius: '12px',
            padding: '16px',
            border: `1px solid ${colors.terracottaLight}`
          }}>
            <TextInput
              label="Medication name"
              placeholder="Enter medication name"
              value={currentMed.name}
              onChange={v => setCurrentMed(p => ({ ...p, name: v }))}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <TextInput
                label="Dosage"
                placeholder="e.g., 500mg"
                value={currentMed.dosage}
                onChange={v => setCurrentMed(p => ({ ...p, dosage: v }))}
              />
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '14px',
                  color: colors.softBrown,
                  marginBottom: '6px'
                }}>
                  Frequency
                </label>
                <select
                  value={currentMed.frequency}
                  onChange={e => setCurrentMed(p => ({ ...p, frequency: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '12px',
                    border: `2px solid ${colors.terracottaLight}`,
                    fontSize: '15px',
                    background: colors.warmWhite
                  }}
                >
                  <option value="">Select</option>
                  <option>Once daily</option>
                  <option>Twice daily</option>
                  <option>3x daily</option>
                  <option>As needed</option>
                </select>
              </div>
            </div>
            <TimeSelect
              label="Time"
              value={currentMed.time}
              onChange={v => setCurrentMed(p => ({ ...p, time: v }))}
            />
            <TextInput
              label="Pill appearance"
              placeholder="e.g., Blue oval"
              value={currentMed.color}
              onChange={v => setCurrentMed(p => ({ ...p, color: v }))}
              helper="Track if pills change"
            />
            <button
              onClick={addMed}
              disabled={!currentMed.name}
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '10px',
                border: 'none',
                background: currentMed.name ? colors.sage : colors.terracottaLight,
                color: colors.warmWhite,
                fontSize: '15px',
                cursor: currentMed.name ? 'pointer' : 'not-allowed',
              }}
            >
              + Add Medication
            </button>
          </div>
        </>
      )
    },
    {
      title: "Your Challenges",
      subtitle: "What do you struggle with most? (optional)",
      required: false,
      content: (
        <MultiSelectInput
          options={[
            'Tracking medications',
            'Remembering appointments',
            'Coordinating with family',
            'Finding reliable help',
            'Managing multiple people',
            'Emergency preparedness'
          ]}
          value={data.challenges}
          onChange={v => update('challenges', v)}
        />
      )
    },
  ] : [
    // Elderly Steps
    {
      title: "Let's get to know you",
      subtitle: "Basic information",
      required: true,
      content: (
        <>
          <TextInput
            label="Your full name"
            placeholder="Enter your full name"
            value={data.name}
            onChange={v => update('name', v)}
            required
            error={errors.name}
          />
          <TextInput
            label="Date of birth"
            placeholder="Enter your date of birth"
            value={data.dateOfBirth}
            onChange={v => update('dateOfBirth', v)}
            required
            error={errors.dateOfBirth}
          />
          <div>
            <label style={{
              display: 'block',
              fontSize: '14px',
              color: colors.softBrown,
              marginBottom: '8px'
            }}>
              Blood type
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-', 'Unknown'].map(bt => (
                <button
                  key={bt}
                  onClick={() => update('bloodType', bt)}
                  style={{
                    padding: '8px 14px',
                    borderRadius: '8px',
                    border: `2px solid ${data.bloodType === bt ? colors.sage : colors.terracottaLight}`,
                    background: data.bloodType === bt ? `${colors.sage}15` : colors.warmWhite,
                    fontSize: '14px',
                    cursor: 'pointer',
                    color: colors.charcoal,
                  }}
                >
                  {bt}
                </button>
              ))}
            </div>
          </div>
        </>
      )
    },
    {
      title: "Emergency Contact",
      subtitle: "Who should we contact?",
      required: true,
      content: (
        <>
          <TextInput
            label="Contact name"
            placeholder="Enter contact's full name"
            value={data.emergencyContact.name}
            onChange={v => updateNested('emergencyContact', 'name', v)}
            required
            error={errors['emergencyContact.name']}
          />
          <TextInput
            label="Relationship"
            placeholder="e.g., Daughter, Son, Neighbor"
            value={data.emergencyContact.relationship}
            onChange={v => updateNested('emergencyContact', 'relationship', v)}
            required
            error={errors['emergencyContact.relationship']}
          />
          <TextInput
            label="Phone number"
            placeholder="Enter their phone number"
            type="tel"
            value={data.emergencyContact.phone}
            onChange={v => updateNested('emergencyContact', 'phone', v)}
            required
            error={errors['emergencyContact.phone']}
          />
        </>
      )
    },
    {
      title: "Your Doctor",
      subtitle: "Primary care physician (optional)",
      required: false,
      content: (
        <>
          <TextInput
            label="Doctor's name"
            placeholder="Enter doctor's name"
            value={data.doctor.name}
            onChange={v => updateNested('doctor', 'name', v)}
          />
          <TextInput
            label="Clinic/Hospital"
            placeholder="Enter clinic name"
            value={data.doctor.clinic}
            onChange={v => updateNested('doctor', 'clinic', v)}
          />
          <TextInput
            label="Phone"
            placeholder="Enter clinic phone"
            type="tel"
            value={data.doctor.phone}
            onChange={v => updateNested('doctor', 'phone', v)}
          />
        </>
      )
    },
    {
      title: "Allergies",
      subtitle: "Select all that apply (optional)",
      required: false,
      content: (
        <>
          <MultiSelectInput
            options={['Penicillin', 'Sulfa drugs', 'Aspirin', 'Ibuprofen', 'Shellfish', 'Peanuts', 'No known allergies']}
            value={data.allergies}
            onChange={v => update('allergies', v)}
          />
          <div style={{ marginTop: '16px' }}>
            <TextInput
              label="Other allergies"
              placeholder="Enter any other allergies"
              value={data.customAllergy}
              onChange={v => update('customAllergy', v)}
            />
          </div>
        </>
      )
    },
    {
      title: "Medical Conditions",
      subtitle: "Current health conditions (optional)",
      required: false,
      content: (
        <MultiSelectInput
          options={['High Blood Pressure', 'Type 2 Diabetes', 'Heart Disease', 'Arthritis', 'COPD', 'Kidney Disease', 'Depression', 'None']}
          value={data.conditions}
          onChange={v => update('conditions', v)}
        />
      )
    },
    {
      title: "Medications",
      subtitle: "What do you take daily? (optional)",
      required: false,
      content: (
        <>
          {data.medications.length > 0 && (
            <div style={{ marginBottom: '16px' }}>
              {data.medications.map(med => (
                <div key={med.id} style={{
                  background: `${colors.sage}10`,
                  borderRadius: '10px',
                  padding: '12px',
                  marginBottom: '8px',
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <div>
                    <p style={{
                      margin: '0 0 2px',
                      fontSize: '15px',
                      fontWeight: '500',
                      color: colors.charcoal
                    }}>
                      {med.name} {med.dosage}
                    </p>
                    <p style={{
                      margin: 0,
                      fontSize: '13px',
                      color: colors.softBrown
                    }}>
                      {med.frequency} {med.time && `at ${med.time}`} {med.color && `• ${med.color}`}
                    </p>
                  </div>
                  <button
                    onClick={() => update('medications', data.medications.filter(m => m.id !== med.id))}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: colors.terracotta,
                      cursor: 'pointer'
                    }}
                  >
                    <Icons.Trash />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div style={{
            background: colors.warmWhite,
            borderRadius: '12px',
            padding: '16px',
            border: `1px solid ${colors.terracottaLight}`
          }}>
            <TextInput
              label="Medication name"
              placeholder="Enter medication name"
              value={currentMed.name}
              onChange={v => setCurrentMed(p => ({ ...p, name: v }))}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <TextInput
                label="Dosage"
                placeholder="e.g., 500mg"
                value={currentMed.dosage}
                onChange={v => setCurrentMed(p => ({ ...p, dosage: v }))}
              />
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '14px',
                  color: colors.softBrown,
                  marginBottom: '6px'
                }}>
                  Frequency
                </label>
                <select
                  value={currentMed.frequency}
                  onChange={e => setCurrentMed(p => ({ ...p, frequency: e.target.value }))}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '12px',
                    border: `2px solid ${colors.terracottaLight}`,
                    fontSize: '15px',
                    background: colors.warmWhite
                  }}
                >
                  <option value="">Select</option>
                  <option>Once daily</option>
                  <option>Twice daily</option>
                  <option>3x daily</option>
                  <option>As needed</option>
                </select>
              </div>
            </div>
            <TimeSelect
              label="Time"
              value={currentMed.time}
              onChange={v => setCurrentMed(p => ({ ...p, time: v }))}
            />
            <TextInput
              label="Pill appearance"
              placeholder="e.g., Small white round"
              value={currentMed.color}
              onChange={v => setCurrentMed(p => ({ ...p, color: v }))}
            />
            <button
              onClick={addMed}
              disabled={!currentMed.name}
              style={{
                width: '100%',
                padding: '12px',
                borderRadius: '10px',
                border: 'none',
                background: currentMed.name ? colors.sage : colors.terracottaLight,
                color: colors.warmWhite,
                fontSize: '15px',
                cursor: currentMed.name ? 'pointer' : 'not-allowed',
              }}
            >
              + Add Medication
            </button>
          </div>
        </>
      )
    },
    {
      title: "Daily Reminders",
      subtitle: "Stay on track (optional)",
      required: false,
      content: (
        <>
          <div style={{ marginBottom: '24px' }}>
            <p style={{
              fontSize: '16px',
              color: colors.charcoal,
              marginBottom: '12px'
            }}>
              Water reminders?
            </p>
            <YesNoInput
              value={data.waterReminder}
              onChange={v => update('waterReminder', v)}
            />
            {data.waterReminder === 'Yes' && (
              <div style={{
                marginTop: '16px',
                padding: '12px',
                background: `${colors.sage}10`,
                borderRadius: '10px'
              }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {['Every 2 hrs', 'Every 3 hrs', '4x daily'].map(f => (
                    <button
                      key={f}
                      onClick={() => update('waterFrequency', f)}
                      style={{
                        padding: '8px 14px',
                        borderRadius: '8px',
                        border: `2px solid ${data.waterFrequency === f ? colors.sage : colors.terracottaLight}`,
                        background: data.waterFrequency === f ? `${colors.sage}20` : colors.warmWhite,
                        fontSize: '14px',
                        cursor: 'pointer',
                      }}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
          <div style={{ marginBottom: '24px' }}>
            <p style={{
              fontSize: '16px',
              color: colors.charcoal,
              marginBottom: '12px'
            }}>
              Medication reminders?
            </p>
            <YesNoInput
              value={data.medicationReminder}
              onChange={v => update('medicationReminder', v)}
            />
          </div>
          <TimeSelect
            label="Wake time"
            value={data.wakeTime}
            onChange={v => update('wakeTime', v)}
          />
          <TimeSelect
            label="Sleep time"
            value={data.sleepTime}
            onChange={v => update('sleepTime', v)}
          />
        </>
      )
    },
  ];

  const progress = ((step + 1) / steps.length) * 100;

  return (
    <div style={{
      minHeight: '100vh',
      background: colors.cream,
      fontFamily: "'Georgia', serif"
    }}>
      <div style={{
        padding: '20px 20px 120px',
        maxWidth: '500px',
        margin: '0 auto'
      }}>
        {/* Progress */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{
            height: '4px',
            background: colors.terracottaLight,
            borderRadius: '2px'
          }}>
            <div style={{
              height: '100%',
              width: `${progress}%`,
              background: colors.sage,
              borderRadius: '2px',
              transition: 'width 0.3s'
            }} />
          </div>
          <p style={{
            fontSize: '13px',
            color: colors.softBrown,
            margin: '8px 0 0',
            textAlign: 'right'
          }}>
            Step {step + 1} of {steps.length}
          </p>
        </div>

        <h2 style={{
          fontSize: 'clamp(22px, 5vw, 26px)',
          fontWeight: '400',
          color: colors.charcoal,
          margin: '0 0 6px'
        }}>
          {steps[step].title}
        </h2>
        <p style={{
          fontSize: '14px',
          color: colors.softBrown,
          margin: '0 0 24px'
        }}>
          {steps[step].subtitle}
        </p>

        {steps[step].content}
      </div>

      {/* Fixed Bottom Navigation */}
      <div style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        background: colors.cream,
        padding: '16px 20px max(20px, env(safe-area-inset-bottom))',
        display: 'flex',
        gap: '12px',
        maxWidth: '500px',
        margin: '0 auto',
        borderTop: `1px solid ${colors.terracottaLight}40`,
      }}>
        {step > 0 && (
          <button
            onClick={() => setStep(step - 1)}
            style={{
              padding: '14px 20px',
              borderRadius: '12px',
              border: `2px solid ${colors.terracottaLight}`,
              background: 'transparent',
              fontSize: '15px',
              color: colors.charcoal,
              cursor: 'pointer',
            }}
          >
            Back
          </button>
        )}
        <button
          onClick={handleContinue}
          style={{
            flex: 1,
            padding: '14px',
            borderRadius: '12px',
            border: 'none',
            background: colors.sage,
            fontSize: '16px',
            color: colors.warmWhite,
            cursor: 'pointer',
          }}
        >
          {step < steps.length - 1 ? 'Continue' : 'Finish Setup'}
        </button>
        {step < steps.length - 1 && canSkip() && (
          <button
            onClick={() => setStep(step + 1)}
            style={{
              padding: '14px 20px',
              borderRadius: '12px',
              border: 'none',
              background: 'transparent',
              fontSize: '14px',
              color: colors.softBrown,
              cursor: 'pointer',
            }}
          >
            Skip
          </button>
        )}
      </div>
    </div>
  );
};

export default Onboarding;
