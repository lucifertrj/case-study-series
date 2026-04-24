const API_BASE_URL = 'http://localhost:8000';

export const savePreferences = async (userType, preferences) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/preferences`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userType,
        preferences,
        createdAt: new Date().toISOString(),
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to save preferences');
    }

    return await response.json();
  } catch (error) {
    console.error('Error saving preferences:', error);
    throw error;
  }
};

export const getPreferences = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/preferences`);

    if (!response.ok) {
      throw new Error('Failed to get preferences');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting preferences:', error);
    throw error;
  }
};
