import React, { useState } from 'react';
import { apiUrl } from '../api';
import './OnboardingModal.css';

const OnboardingModal = ({
  isOpen,
  onClose,
  onComplete,
  onProcessingStart,
  onProcessingError,
  userId,
  isMandatory = false,
}) => {
  const [step, setStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [preferences, setPreferences] = useState({
    travel_style: [],
    property_preferences: [],
    typical_budget: '',
    must_have_amenities: [],
    preferred_vibes: [],
    typical_companions: [],
    accessibility_needs: [],
    location_preferences: [],
    custom_interests: '',
    dietary_preferences: [],
  });

  if (!isOpen) return null;

  const toggleArrayItem = (field, value) => {
    if (field === 'dietary_preferences') {
      setPreferences((prev) => {
        const currentValues = prev[field];
        if (value === 'no preference') {
          return { ...prev, [field]: currentValues.includes(value) ? [] : [value] };
        }

        const withoutNoPreference = currentValues.filter((v) => v !== 'no preference');
        return {
          ...prev,
          [field]: withoutNoPreference.includes(value)
            ? withoutNoPreference.filter((v) => v !== value)
            : [...withoutNoPreference, value],
        };
      });
      return;
    }

    setPreferences((prev) => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter((v) => v !== value)
        : [...prev[field], value],
    }));
  };

  const handleSubmit = async () => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    onProcessingStart?.();

    try {
      const res = await fetch(apiUrl('/onboarding'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          ...preferences,
        }),
      });

      if (!res.ok) throw new Error('Failed to save preferences');

      const data = await res.json();
      onComplete(data);
      onClose();
    } catch (error) {
      onProcessingError?.();
      setIsSubmitting(false);
      alert('Failed to save preferences. Please try again.');
    }
  };

  const renderStep1 = () => (
    <div className="onboarding-step">
      <h3>What's your travel style?</h3>
      <p className="step-description">Select all that apply</p>
      <div className="options-grid">
        {['Adventure', 'Relaxation', 'Cultural', 'Foodie', 'Party', 'Wellness', 'Nature', 'Shopping'].map((style) => (
          <button
            key={style}
            className={`option-btn ${preferences.travel_style.includes(style.toLowerCase()) ? 'selected' : ''}`}
            onClick={() => toggleArrayItem('travel_style', style.toLowerCase())}
          >
            {style}
          </button>
        ))}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="onboarding-step">
      <h3>What type of properties do you prefer?</h3>
      <p className="step-description">Select your favorites</p>
      <div className="options-grid">
        {['Hotel', 'Boutique', 'Resort', 'Villa', 'Apartment', 'Hostel', 'B&B', 'Ryokan'].map((type) => (
          <button
            key={type}
            className={`option-btn ${preferences.property_preferences.includes(type.toLowerCase()) ? 'selected' : ''}`}
            onClick={() => toggleArrayItem('property_preferences', type.toLowerCase())}
          >
            {type}
          </button>
        ))}
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="onboarding-step">
      <h3>What's your typical budget?</h3>
      <p className="step-description">Choose one</p>
      <div className="options-vertical">
        {[
          { value: 'budget', label: 'Budget ($0-150/night)', emoji: '💵' },
          { value: 'mid-range', label: 'Mid-range ($100-300/night)', emoji: '💳' },
          { value: 'luxury', label: 'Luxury ($250+/night)', emoji: '💎' },
        ].map((budget) => (
          <button
            key={budget.value}
            className={`option-btn large ${preferences.typical_budget === budget.value ? 'selected' : ''}`}
            onClick={() => setPreferences((prev) => ({ ...prev, typical_budget: budget.value }))}
          >
            <span className="option-emoji">{budget.emoji}</span>
            <span>{budget.label}</span>
          </button>
        ))}
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="onboarding-step">
      <h3>Must-have amenities?</h3>
      <p className="step-description">Select your essentials</p>
      <div className="options-grid">
        {['WiFi', 'Pool', 'Gym', 'Spa', 'Parking', 'Restaurant', 'Bar', 'Room Service', 'Breakfast', 'Pet-friendly'].map(
          (amenity) => (
            <button
              key={amenity}
              className={`option-btn ${preferences.must_have_amenities.includes(amenity.toLowerCase()) ? 'selected' : ''
                }`}
              onClick={() => toggleArrayItem('must_have_amenities', amenity.toLowerCase())}
            >
              {amenity}
            </button>
          )
        )}
      </div>
    </div>
  );

  const renderStep5 = () => (
    <div className="onboarding-step">
      <h3>What atmosphere do you enjoy?</h3>
      <p className="step-description">Choose your vibe</p>
      <div className="options-grid">
        {['Romantic', 'Quiet', 'Social', 'Trendy', 'Family-friendly', 'Modern', 'Historic', 'Cozy'].map((vibe) => (
          <button
            key={vibe}
            className={`option-btn ${preferences.preferred_vibes.includes(vibe.toLowerCase()) ? 'selected' : ''}`}
            onClick={() => toggleArrayItem('preferred_vibes', vibe.toLowerCase())}
          >
            {vibe}
          </button>
        ))}
      </div>
    </div>
  );

  const renderStep6 = () => (
    <div className="onboarding-step">
      <h3>Who do you usually travel with?</h3>
      <p className="step-description">Select all that apply</p>
      <div className="options-grid">
        {['Solo', 'Couple', 'Family', 'Friends', 'Business'].map((companion) => (
          <button
            key={companion}
            className={`option-btn ${preferences.typical_companions.includes(companion.toLowerCase()) ? 'selected' : ''
              }`}
            onClick={() => toggleArrayItem('typical_companions', companion.toLowerCase())}
          >
            {companion}
          </button>
        ))}
      </div>
    </div>
  );

  const renderStep7 = () => (
    <div className="onboarding-step">
      <h3>Preferred locations?</h3>
      <p className="step-description">What kind of locations do you love?</p>
      <div className="options-grid">
        {['Beach', 'City Center', 'Mountains', 'Countryside', 'Desert', 'Lake', 'Historic District', 'Nightlife Area'].map(
          (location) => (
            <button
              key={location}
              className={`option-btn ${preferences.location_preferences.includes(location.toLowerCase()) ? 'selected' : ''
                }`}
              onClick={() => toggleArrayItem('location_preferences', location.toLowerCase())}
            >
              {location}
            </button>
          )
        )}
      </div>
    </div>
  );

  const renderStep8 = () => (
    <div className="onboarding-step">
      <h3>Any custom interests?</h3>
      <p className="step-description">Optional, but useful for itinerary planning</p>
      <textarea
        className="onboarding-textarea"
        value={preferences.custom_interests}
        onChange={(e) => setPreferences((prev) => ({ ...prev, custom_interests: e.target.value }))}
        placeholder="Art museums, food tours, live music, architecture walks, sunrise hikes..."
        rows={5}
      />
    </div>
  );

  const renderStep9 = () => (
    <div className="onboarding-step">
      <h3>Any dietary preferences?</h3>
      <p className="step-description">These help the itinerary planner suggest better food stops</p>
      <div className="options-grid">
        {['Vegetarian', 'Vegan', 'Halal', 'Kosher', 'Gluten-free', 'Dairy-free', 'Nut-free', 'No preference'].map(
          (diet) => (
            <button
              key={diet}
              className={`option-btn ${preferences.dietary_preferences.includes(diet.toLowerCase()) ? 'selected' : ''}`}
              onClick={() => toggleArrayItem('dietary_preferences', diet.toLowerCase())}
            >
              {diet}
            </button>
          )
        )}
      </div>
    </div>
  );

  const totalSteps = 9;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h2>Welcome! Let's personalize your experience</h2>
          {!isMandatory && (
            <button className="close-btn" onClick={onClose}>
              ×
            </button>
          )}
        </div>

        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${(step / totalSteps) * 100}%` }}></div>
        </div>

        <div className="modal-body">
          {step === 1 && renderStep1()}
          {step === 2 && renderStep2()}
          {step === 3 && renderStep3()}
          {step === 4 && renderStep4()}
          {step === 5 && renderStep5()}
          {step === 6 && renderStep6()}
          {step === 7 && renderStep7()}
          {step === 8 && renderStep8()}
          {step === 9 && renderStep9()}
        </div>

        <div className="modal-footer">
          <div className="step-indicator">
            Step {step} of {totalSteps}
          </div>
          <div className="modal-actions">
            {step > 1 && (
              <button className="btn-secondary" onClick={() => setStep(step - 1)} disabled={isSubmitting}>
                Back
              </button>
            )}
            {step < totalSteps ? (
              <button className="btn-primary" onClick={() => setStep(step + 1)} disabled={isSubmitting}>
                Next
              </button>
            ) : (
              <button className="btn-primary" onClick={handleSubmit} disabled={isSubmitting}>
                {isSubmitting ? 'Setting up memory...' : 'Complete Setup'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingModal;
