def format_profile_for_cognee(user_type: str, data: dict) -> str:
    """Format user profile data for Cognee ingestion."""
    if user_type == "caregiver":
        profile = f"""
CAREGIVER PROFILE: {data.get('name', 'Unknown')}
Phone: {data.get('phone', 'Not provided')}

CARE RECIPIENT: {data.get('currentRecipient', {}).get('name', 'Unknown')}
Relationship: {data.get('currentRecipient', {}).get('relationship', 'Not specified')}
DOB: {data.get('currentRecipient', {}).get('dateOfBirth', 'Not provided')}
Phone: {data.get('currentRecipient', {}).get('phone', 'Not provided')}
Address: {data.get('currentRecipient', {}).get('address', 'Not provided')}

DOCTOR: {data.get('doctor', {}).get('name', 'Not provided')}
Clinic: {data.get('doctor', {}).get('clinic', 'Not provided')}
Doctor Phone: {data.get('doctor', {}).get('phone', 'Not provided')}

Blood Type: {data.get('bloodType', 'Unknown')}

ALLERGIES: {', '.join(data.get('allergies', [])) or 'None recorded'}
{f"Other allergies: {data.get('customAllergy')}" if data.get('customAllergy') else ''}

MEDICAL CONDITIONS: {', '.join(data.get('conditions', [])) or 'None recorded'}

MEDICATIONS:
"""
        medications = data.get("medications", [])
        if medications:
            for med in medications:
                profile += f"- {med.get('name', 'Unknown')} {med.get('dosage', '')} | {med.get('frequency', '')} at {med.get('time', 'unspecified time')} | Appearance: {med.get('color', 'not recorded')}\n"
        else:
            profile += "No medications recorded\n"

        profile += f"""
CAREGIVER CHALLENGES: {', '.join(data.get('challenges', [])) or 'None specified'}
"""
    else:
        # Elderly user profile
        profile = f"""
CARE PROFILE: {data.get('name', 'Unknown')}
DOB: {data.get('dateOfBirth', 'Not provided')}
Blood Type: {data.get('bloodType', 'Unknown')}

EMERGENCY CONTACT: {data.get('emergencyContact', {}).get('name', 'Not provided')}
Relationship: {data.get('emergencyContact', {}).get('relationship', 'Not specified')}
Phone: {data.get('emergencyContact', {}).get('phone', 'Not provided')}

DOCTOR: {data.get('doctor', {}).get('name', 'Not provided')}
Clinic: {data.get('doctor', {}).get('clinic', 'Not provided')}
Doctor Phone: {data.get('doctor', {}).get('phone', 'Not provided')}

ALLERGIES: {', '.join(data.get('allergies', [])) or 'None recorded'}
{f"Other allergies: {data.get('customAllergy')}" if data.get('customAllergy') else ''}

MEDICAL CONDITIONS: {', '.join(data.get('conditions', [])) or 'None recorded'}

MEDICATIONS:
"""
        medications = data.get("medications", [])
        if medications:
            for med in medications:
                profile += f"- {med.get('name', 'Unknown')} {med.get('dosage', '')} | {med.get('frequency', '')} at {med.get('time', 'unspecified time')} | Appearance: {med.get('color', 'not recorded')}\n"
        else:
            profile += "No medications recorded\n"

        profile += f"""
DAILY REMINDERS:
Water Reminders: {data.get('waterReminder', 'Not set')} {f"({data.get('waterFrequency')})" if data.get('waterFrequency') else ''}
Medication Reminders: {data.get('medicationReminder', 'Not set')}
Wake Time: {data.get('wakeTime', 'Not set')}
Sleep Time: {data.get('sleepTime', 'Not set')}
"""

    return profile.strip()


def format_log_entry(log_type: str, title: str, details: str | None, timestamp: str) -> str:
    """Format a log entry for Cognee ingestion."""
    return f"""
[{log_type.upper()}] {title}
{details or ''}
Logged at: {timestamp}
""".strip()
