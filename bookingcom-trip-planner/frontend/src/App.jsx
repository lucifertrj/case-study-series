import React, { useEffect, useState } from 'react';
import PropertyCard from './components/PropertyCard';
import OnboardingModal from './components/OnboardingModal';
import ItineraryPlanner from './components/ItineraryPlanner';
import { apiUrl } from './api';
import './index.css';

function App() {
  const MIN_MEMORY_SETUP_MS = 1800;
  const [appStage, setAppStage] = useState('welcome');
  const [activePanel, setActivePanel] = useState('search');
  const [welcomeUserIdInput, setWelcomeUserIdInput] = useState('');
  const [currentUserId, setCurrentUserId] = useState('');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [errorToast, setErrorToast] = useState('');
  const [errorHtml, setErrorHtml] = useState(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [hasPreferences, setHasPreferences] = useState(false);
  const [preferenceSummary, setPreferenceSummary] = useState('');
  const [userPreferences, setUserPreferences] = useState(null);
  const [memorySetupStartedAt, setMemorySetupStartedAt] = useState(null);
  const [memorySetupReturnStage, setMemorySetupReturnStage] = useState('home');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(apiUrl('/health'));
        const result = await res.json();
        if (result.needs_indexing) {
          showError('Qdrant has 0 properties. Run: python -m scripts.index_properties', true);
        }
      } catch (e) {
        // Backend not up yet.
      }
    };
    checkHealth();
  }, []);

  const showError = (msg, showInResults = false) => {
    setErrorToast(`⚠️ ${msg}`);
    setTimeout(() => setErrorToast(''), 5000);

    if (showInResults) {
      setErrorHtml(
        <div className="empty-state">
          <div className="empty-icon">⚠️</div>
          <div className="empty-title">Backend not reachable</div>
          <div className="empty-text">
            Start the backend: <code>cd backend && uvicorn main:app --reload</code>
            <br />
            Then index properties: <code>python -m scripts.index_properties</code>
          </div>
        </div>
      );
    }
  };

  const handleWelcomeSubmit = async (e) => {
    e.preventDefault();
    if (!welcomeUserIdInput.trim()) return;

    const nextUserId = welcomeUserIdInput.trim();
    setLoading(true);
    setCurrentUserId(nextUserId);

    try {
      const res = await fetch(apiUrl(`/preferences/${nextUserId}`));
      if (!res.ok) throw new Error('Failed to fetch preferences');

      const result = await res.json();
      setHasPreferences(result.has_preferences);
      setPreferenceSummary(result.summary || '');
      setUserPreferences(result.preferences || null);
      setAppStage(result.has_preferences ? 'home' : 'onboarding');
    } catch (e) {
      setUserPreferences(null);
      setAppStage('onboarding');
    } finally {
      setLoading(false);
    }
  };

  const sendSignal = async (type, propData) => {
    if (!currentUserId) return;
    try {
      await fetch(apiUrl('/signal'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId,
          session_id: 'session_1',
          property_id: propData.id || '',
          signal_type: type,
          property_data: propData,
        }),
      });
    } catch (e) {
      // Non-critical.
    }
  };

  const handleCardClick = (property) => {
    sendSignal('click', property);
  };

  const handleBook = (e, property) => {
    e.stopPropagation();
    sendSignal('book', property);

    const btn = e.target;
    btn.textContent = '✓ Booked!';
    btn.style.background = '#1a9e8f';
    setTimeout(() => {
      btn.textContent = 'Book →';
      btn.style.background = '';
    }, 2500);
  };

  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setErrorHtml(null);

    try {
      const res = await fetch(apiUrl('/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          user_id: currentUserId || 'anonymous',
          session_id: 'session_1',
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const resData = await res.json();
      setData(resData);
      setActivePanel('search');
    } catch (e) {
      const isNetworkError = e instanceof TypeError;
      showError(e.message || 'Search failed.', isNetworkError);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleChipClick = (text) => {
    setQuery(text);
    setTimeout(() => {
      document.getElementById('searchForm')?.requestSubmit();
    }, 0);
  };

  const handleOnboardingComplete = (result) => {
    setHasPreferences(true);
    setPreferenceSummary(result.preferences_summary);
    setUserPreferences(result.preferences || null);
    setShowOnboarding(false);
    const elapsed = memorySetupStartedAt ? Date.now() - memorySetupStartedAt : 0;
    const remaining = Math.max(0, MIN_MEMORY_SETUP_MS - elapsed);
    setTimeout(() => {
      setAppStage('home');
      setErrorToast('✓ Your travel memory has been saved!');
      setTimeout(() => setErrorToast(''), 3000);
      setMemorySetupStartedAt(null);
    }, remaining);
  };

  const handleOnboardingProcessingStart = () => {
    setMemorySetupReturnStage(appStage);
    setShowOnboarding(false);
    setMemorySetupStartedAt(Date.now());
    setAppStage('saving_memory');
  };

  const handleOnboardingProcessingError = () => {
    const previousStage = memorySetupReturnStage || 'home';
    setAppStage(previousStage);
    setMemorySetupStartedAt(null);
    if (previousStage === 'home') {
      setShowOnboarding(true);
    }
  };

  const handleStartOnboarding = () => {
    if (!currentUserId || currentUserId === 'anonymous') {
      alert('Please enter a user ID first to save your preferences!');
      document.getElementById('userIdInput')?.focus();
      return;
    }
    setShowOnboarding(true);
  };

  const renderWelcome = () => (
    <div className="welcome">
      <div className="how-it-works">
        <div className="how-step">
          <div className="step-num">1</div>
          <div className="step-title">Intent Parser</div>
          <div className="step-text">
            GPT-4o-mini extracts structured filters from your natural language query — including
            implicit amenities you didn&apos;t explicitly state.
          </div>
        </div>
        <div className="how-step">
          <div className="step-num">2</div>
          <div className="step-title">Memory Enricher</div>
          <div className="step-text">
            Retrieves your travel DNA from Cognee and merges it with your current intent.
          </div>
        </div>
        <div className="how-step">
          <div className="step-num">3</div>
          <div className="step-title">Semantic Ranker</div>
          <div className="step-text">
            Runs hybrid search so “sunset from balcony” can match your “scenic views” intent.
          </div>
        </div>
        <div className="how-step">
          <div className="step-num">4</div>
          <div className="step-title">Composite Scorer</div>
          <div className="step-text">
            Blends semantic relevance, filters, quality, and memory alignment into the final ranking.
          </div>
        </div>
      </div>
    </div>
  );

  const renderIntentPills = (intent) => {
    if (!intent) return null;
    return (
      <div className="intent-strip">
        {intent.vibe?.map((v, i) => <span key={`vibe-${i}`} className="intent-pill vibe">✨ {v}</span>)}
        {intent.amenities?.map((a, i) => <span key={`amenity-${i}`} className="intent-pill amenity">🏊 {a}</span>)}
        {intent.property_type?.map((t, i) => <span key={`type-${i}`} className="intent-pill type">🏨 {t}</span>)}
        {intent.destination && <span className="intent-pill loc">📍 {intent.destination}</span>}
        {intent.location_vibe && <span className="intent-pill loc">🌊 {intent.location_vibe}</span>}
        {intent.budget_range && (
          <span className="intent-pill budget">
            💰 ${intent.budget_range[0]}–${intent.budget_range[1]}/night
          </span>
        )}
      </div>
    );
  };

  const renderContent = () => {
    if (errorHtml && !data) return errorHtml;
    if (!data) return renderWelcome();

    if (!data.results || data.results.length === 0) {
      return (
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <div className="empty-title">No matches found</div>
          <div className="empty-text">
            Try a different query — or check the backend is running and properties are indexed.
          </div>
        </div>
      );
    }

    const intent = data.parsed_intent;

    return (
      <>
        <div className="results-header">
          <div>
            <div className="results-title">{data.results.length} Properties Found</div>
            <div className="results-count">
              For: <em>&quot;{data.query}&quot;</em>
            </div>
          </div>
        </div>

        {renderIntentPills(intent)}

        {data.memory_used && (
          <div className="memory-banner">
            <span className="memory-icon">🧠</span>
            <div>
              <strong>Personalised for you</strong> — {data.memory_summary || 'Results enriched with your travel history.'}
            </div>
          </div>
        )}

        <div className="cards-grid">
          {data.results.map((property, i) => (
            <div
              key={property.id}
              style={{ animationDelay: `${i * 0.06}s` }}
              className="card-anim-wrapper"
            >
              <PropertyCard
                property={property}
                intent={intent}
                currentUserId={currentUserId}
                onClick={handleCardClick}
                onBook={handleBook}
              />
            </div>
          ))}
        </div>

        <div className="planner-cta">
          <div className="planner-cta-copy">
            <div className="planner-cta-eyebrow">Next step</div>
            <div className="planner-cta-title">Turn these hotel matches into a full itinerary</div>
            <div className="planner-cta-text">
              Carry over the latest search intent, extracted memory context, and onboarding preferences into the itinerary planner chat.
            </div>
          </div>
          <button className="planner-cta-btn" onClick={() => setActivePanel('planner')}>
            Open itinerary planner
          </button>
        </div>
      </>
    );
  };

  return (
    <>
      <OnboardingModal
        isOpen={appStage === 'onboarding' || showOnboarding}
        onClose={() => {
          if (appStage !== 'onboarding') {
            setShowOnboarding(false);
          }
        }}
        onComplete={handleOnboardingComplete}
        onProcessingStart={handleOnboardingProcessingStart}
        onProcessingError={handleOnboardingProcessingError}
        userId={currentUserId}
        isMandatory={appStage === 'onboarding'}
      />

      {appStage === 'welcome' && (
        <section className="hero welcome-stage">
          <div className="hero-noise"></div>
          <div className="hero-inner welcome-stage-inner">
            <div className="hero-badge">
              <div className="dot"></div>
              AI Smart Filter
            </div>
            <h1>
              Welcome to <br />
              <em>Tailored Travel</em>
            </h1>
            <p className="welcome-subtitle">
              Enter your User ID to load your travel memory, or set up a new profile.
            </p>
            <form className="welcome-form" onSubmit={handleWelcomeSubmit}>
              <input
                type="text"
                id="userIdInput"
                placeholder="Enter User ID..."
                value={welcomeUserIdInput}
                onChange={(e) => setWelcomeUserIdInput(e.target.value)}
                autoFocus
                required
                className="welcome-input"
              />
              <button type="submit" className="search-btn welcome-btn" disabled={loading}>
                {loading ? 'Loading...' : 'Continue'}
              </button>
            </form>
          </div>
        </section>
      )}

      {appStage === 'saving_memory' && (
        <section className="hero saving-memory-stage">
          <div className="hero-noise"></div>
          <div className="saving-container">
            <div className="memory-setup-visual" aria-hidden="true">
              <span className="memory-ring outer"></span>
              <span className="memory-ring mid"></span>
              <span className="memory-ring inner"></span>
              <span className="memory-core"></span>
              <span className="memory-pulse p1"></span>
              <span className="memory-pulse p2"></span>
              <span className="memory-pulse p3"></span>
            </div>
            <h2>Setting up your travel memory</h2>
            <p>Building your profile graph so every search can use your preferences instantly.</p>
          </div>
        </section>
      )}

      {appStage === 'home' && (
        <section className="hero">
          <div className="hero-noise"></div>
          <div className="hero-inner">
            <div className="user-row">
              <span className="user-label">Logged in as:</span>
              <span className="user-badge" style={{ fontWeight: '600', color: '#111827', marginRight: '16px' }}>
                {currentUserId}
              </span>
              <span
                className="anon-badge"
                style={{
                  background: 'rgba(26,158,143,0.2)',
                  borderColor: 'rgba(26,158,143,0.4)',
                  color: '#80d5cc',
                }}
              >
                🧠 Memory ON
              </span>
              {hasPreferences ? (
                <button className="onboarding-btn preferences-set" onClick={handleStartOnboarding}>
                  ✓ Preferences Set
                </button>
              ) : (
                <button className="onboarding-btn" onClick={handleStartOnboarding}>
                  ✨ Set Preferences
                </button>
              )}
            </div>

            {hasPreferences && preferenceSummary && (
              <div className="preferences-banner">
                <span className="pref-icon">✨</span>
                <div className="pref-text">
                  <strong>Your preferences:</strong> {preferenceSummary}
                </div>
              </div>
            )}

            <div className="hero-badge">
              <div className="dot"></div>
              AI Smart Filter — PoC
            </div>

            <h1>
              Find hotels that<br />
              <em>actually</em> get you
            </h1>
            <p>
              Search in plain English. Just type what you want like “sunset views” or “quiet retreat” and let AI handle the rest.
            </p>

            <form id="searchForm" onSubmit={handleSearch}>
              <div className="search-container">
                <div className="search-input-wrap">
                  <svg className="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35" />
                  </svg>
                  <input
                    id="searchInput"
                    type="text"
                    placeholder="Romantic boutique with rooftop views in Paris…"
                    autoComplete="off"
                    autoFocus
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>
                <button className="search-btn" type="submit" disabled={loading}>
                  {loading ? 'Searching…' : 'Search'}
                </button>
              </div>
            </form>

            <div className="suggestions">
              <span className="suggestion-chip" onClick={() => handleChipClick('Sunset views & great gym in Bali')}>
                Sunset views & great gym in Bali
              </span>
              <span className="suggestion-chip" onClick={() => handleChipClick('Romantic villa with private pool')}>
                Romantic villa with private pool
              </span>
              <span className="suggestion-chip" onClick={() => handleChipClick('Family resort with kids club in Tokyo')}>
                Family resort with kids club in Tokyo
              </span>
              <span className="suggestion-chip" onClick={() => handleChipClick('Boutique hotel near beach in Barcelona')}>
                Boutique hotel near beach in Barcelona
              </span>
              <span className="suggestion-chip" onClick={() => handleChipClick('Quiet wellness retreat in Amsterdam')}>
                Quiet wellness retreat in Amsterdam
              </span>
            </div>
          </div>
        </section>
      )}

      {loading && <div id="loadingBar" className="loading-bar"></div>}

      {appStage === 'home' && (
        <main className="main">
          {activePanel === 'planner' && data ? (
            <ItineraryPlanner
              searchData={data}
              currentUserId={currentUserId}
              preferenceSummary={preferenceSummary}
              userPreferences={userPreferences}
              onBack={() => setActivePanel('search')}
            />
          ) : (
            renderContent()
          )}
        </main>
      )}

      {errorToast && <div className="toast">{errorToast}</div>}
    </>
  );
}

export default App;
