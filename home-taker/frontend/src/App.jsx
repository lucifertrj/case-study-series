import React, { useState } from 'react';
import { colors } from './styles/colors';
import SplashScreen from './screens/SplashScreen';
import UserTypeScreen from './screens/UserTypeScreen';
import Onboarding from './screens/Onboarding';
import Dashboard from './screens/Dashboard';
import Schedule from './screens/Schedule';
import Log from './screens/Log';
import Chat from './screens/Chat';
import SavingOverlay from './components/SavingOverlay';
import { savePreferences } from './services/api';

function App() {
  const [screen, setScreen] = useState('splash');
  const [userType, setUserType] = useState(null);
  const [userData, setUserData] = useState(null);
  const [saving, setSaving] = useState(false);

  const navigate = (s) => setScreen(s);

  const handleUserTypeSelect = (type) => {
    setUserType(type);
    setScreen('onboarding');
  };

  const handleOnboardingComplete = async (data) => {
    setUserData(data);
    setSaving(true);

    try {
      await savePreferences(userType, data);
      console.log('Preferences saved successfully');
    } catch (error) {
      console.error('Failed to save preferences, continuing anyway:', error);
    } finally {
      setSaving(false);
      setScreen('dashboard');
    }
  };

  // Chat screen is full-width, other screens have max-width
  const isFullWidthScreen = screen === 'chat';

  return (
    <div style={{
      maxWidth: isFullWidthScreen ? '100%' : '500px',
      margin: '0 auto',
      minHeight: '100vh',
    }}>
      {screen === 'splash' && (
        <SplashScreen onStart={() => setScreen('userType')} />
      )}
      {
        screen === 'userType' && (
          <UserTypeScreen onSelect={handleUserTypeSelect} />
        )
      }
      {
        screen === 'onboarding' && (
          <Onboarding
            userType={userType}
            onComplete={handleOnboardingComplete}
          />
        )
      }
      {
        screen === 'dashboard' && (
          <Dashboard
            userData={userData}
            userType={userType}
            onNavigate={navigate}
          />
        )
      }
      {
        screen === 'schedule' && (
          <Schedule
            userData={userData}
            userType={userType}
            onNavigate={navigate}
          />
        )
      }
      {
        screen === 'log' && (
          <Log
            userData={userData}
            userType={userType}
            onNavigate={navigate}
          />
        )
      }
      {
        screen === 'chat' && (
          <Chat
            userData={userData}
            userType={userType}
            onNavigate={navigate}
          />
        )
      }

      {/* Loading overlay when saving */}
      {saving && <SavingOverlay message="Saving your preferences..." />}
    </div >
  );
}

export default App;
