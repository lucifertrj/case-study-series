import React, { useEffect, useState } from 'react';
import { apiUrl } from '../api';

const PERSONALIZATION_LOADING_STEPS = [
  'Model is reading your query and preferences',
  'Matching top properties to your travel style',
  'Drafting a personalized day-by-day structure',
];

const ITINERARY_LOADING_STEPS = [
  'Model is planning your next revision',
  'Balancing pacing, activities, and food stops',
  'Finalizing recommendations with your context',
];

const normalizeMarkdownPayload = (content = '') => {
  let normalized = String(content ?? '').replace(/\r\n?/g, '\n');

  // Some model responses return literal escaped newlines/tabs.
  if (normalized.includes('\\n')) {
    normalized = normalized.replace(/\\n/g, '\n');
  }
  if (normalized.includes('\\t')) {
    normalized = normalized.replace(/\\t/g, '  ');
  }

  // If the entire response is wrapped in a markdown fence, unwrap it.
  const fenced = normalized.match(/^```([a-zA-Z0-9_-]+)?\n([\s\S]*?)\n```$/);
  if (fenced) {
    const lang = (fenced[1] || '').toLowerCase();
    if (!lang || lang === 'md' || lang === 'markdown') {
      normalized = fenced[2];
    }
  }

  return normalized.trim();
};

const escapeHtml = (text = '') =>
  text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const formatInlineMarkdown = (line) => {
  let formatted = escapeHtml(line);
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, text, url) => {
    const safeUrl = /^(https?:\/\/)/i.test(url) ? url : '#';
    return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${text}</a>`;
  });
  return formatted;
};

const markdownToHtml = (markdown = '') => {
  const lines = normalizeMarkdownPayload(markdown).split('\n');
  const html = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;

  const closeLists = () => {
    if (inUl) {
      html.push('</ul>');
      inUl = false;
    }
    if (inOl) {
      html.push('</ol>');
      inOl = false;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      closeLists();
      if (!inCode) {
        inCode = true;
        html.push('<pre><code>');
      } else {
        inCode = false;
        html.push('</code></pre>');
      }
      continue;
    }

    if (inCode) {
      html.push(`${escapeHtml(line)}\n`);
      continue;
    }

    if (!trimmed) {
      closeLists();
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeLists();
      const level = heading[1].length;
      html.push(`<h${level}>${formatInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const ulItem = trimmed.match(/^[-*]\s+(.*)$/);
    if (ulItem) {
      if (!inUl) {
        if (inOl) {
          html.push('</ol>');
          inOl = false;
        }
        html.push('<ul>');
        inUl = true;
      }
      html.push(`<li>${formatInlineMarkdown(ulItem[1])}</li>`);
      continue;
    }

    const olItem = trimmed.match(/^\d+\.\s+(.*)$/);
    if (olItem) {
      if (!inOl) {
        if (inUl) {
          html.push('</ul>');
          inUl = false;
        }
        html.push('<ol>');
        inOl = true;
      }
      html.push(`<li>${formatInlineMarkdown(olItem[1])}</li>`);
      continue;
    }

    if (trimmed.startsWith('>')) {
      closeLists();
      html.push(`<blockquote>${formatInlineMarkdown(trimmed.replace(/^>\s?/, ''))}</blockquote>`);
      continue;
    }

    closeLists();
    html.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
  }

  closeLists();
  if (inCode) {
    html.push('</code></pre>');
  }
  return html.join('');
};

const buildInitialPrompt = (searchData, userPreferences) => {
  const destination = searchData?.parsed_intent?.destination;
  const dietary = userPreferences?.dietary_preferences?.length
    ? `Keep these dietary preferences in mind: ${userPreferences.dietary_preferences.join(', ')}.`
    : '';
  const interests = userPreferences?.custom_interests
    ? `Also lean into these custom interests: ${userPreferences.custom_interests}.`
    : '';

  return `Create a practical itinerary${destination ? ` for ${destination}` : ''} based on my latest hotel search. Recommend which shortlisted property fits best, then give me a day-by-day plan with food and activity ideas. ${dietary} ${interests}`.trim();
};

function ItineraryPlanner({
  searchData,
  currentUserId,
  preferenceSummary,
  userPreferences,
  onBack,
}) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState('Building your itinerary...');
  const [loadingStepIndex, setLoadingStepIndex] = useState(0);

  const loadingSteps = loadingLabel.startsWith('Personalizing')
    ? PERSONALIZATION_LOADING_STEPS
    : ITINERARY_LOADING_STEPS;

  const sendMessage = async (messageText, historyOverride = messages) => {
    if (!messageText.trim()) return;

    const nextHistory = [...historyOverride, { role: 'user', content: messageText.trim() }];
    setMessages(nextHistory);
    setLoadingLabel(historyOverride.length === 0 ? 'Personalizing your result...' : 'Building your itinerary...');
    setLoading(true);

    try {
      const res = await fetch(apiUrl('/itinerary-chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: currentUserId || 'anonymous',
          message: messageText.trim(),
          search_query: searchData.query,
          parsed_intent: searchData.parsed_intent || {},
          memory_summary: searchData.memory_summary || null,
          preferences_summary: preferenceSummary || null,
          top_properties: (searchData.results || []).slice(0, 5),
          history: historyOverride,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const result = await res.json();
      setMessages((prev) => [...prev, { role: 'assistant', content: result.reply }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: error.message || 'Failed to generate itinerary.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!searchData || messages.length > 0) return;
    const initialPrompt = buildInitialPrompt(searchData, userPreferences);
    void sendMessage(initialPrompt, []);
  }, [searchData, userPreferences, messages.length]);

  useEffect(() => {
    if (!loading) return undefined;
    setLoadingStepIndex(0);
    const timer = setInterval(() => {
      setLoadingStepIndex((prev) => (prev + 1) % loadingSteps.length);
    }, 1400);
    return () => clearInterval(timer);
  }, [loading, loadingSteps.length]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const message = draft;
    setDraft('');
    await sendMessage(message);
  };

  return (
    <section className="planner-shell">
      <div className="planner-topbar">
        <button className="planner-back-btn" onClick={onBack}>
          ← Back to results
        </button>
        <div className="planner-meta">
          <div className="planner-title">Itinerary Planner</div>
          <div className="planner-subtitle">Grounded in your latest search context and travel memory</div>
        </div>
      </div>

      <div className="planner-context-card">
        <div className="planner-context-label">Search carried into planner</div>
        <div className="planner-context-query">“{searchData.query}”</div>
        {searchData.memory_summary && (
          <div className="planner-context-memory">
            <strong>Memory:</strong> {searchData.memory_summary}
          </div>
        )}
        {preferenceSummary && (
          <div className="planner-context-memory">
            <strong>Preferences:</strong> {preferenceSummary}
          </div>
        )}
      </div>

      <div className="planner-chat">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`planner-message ${message.role === 'user' ? 'planner-message-user' : 'planner-message-assistant'}`}
          >
            <div className="planner-message-role">{message.role === 'user' ? 'You' : 'Planner'}</div>
            {message.role === 'assistant' ? (
              <div
                className="planner-message-body planner-markdown"
                dangerouslySetInnerHTML={{ __html: markdownToHtml(message.content) }}
              />
            ) : (
              <div className="planner-message-body">{message.content}</div>
            )}
          </div>
        ))}

        {loading && (
          <div className="planner-message planner-message-assistant">
            <div className="planner-message-role">Planner</div>
            <div className="planner-message-body planner-loader">
              <div className="planner-loader-spinner" aria-hidden="true" />
              <div>
                <div className="planner-loader-title">
                  {loadingLabel}
                  <span className="planner-loader-dots" aria-hidden="true" />
                </div>
                <div className="planner-loader-status">{loadingSteps[loadingStepIndex]}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <form className="planner-compose" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask for a 3-day plan, a vegetarian version, quieter evenings, local markets, or family pacing…"
          rows={4}
        />
        <button className="search-btn" type="submit" disabled={loading || !draft.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}

export default ItineraryPlanner;
