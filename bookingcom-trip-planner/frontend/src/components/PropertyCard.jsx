import React from 'react';

const PropertyCard = ({ property: p, intent, currentUserId, onBook, onClick }) => {
    const score = Math.round(p.composite_score * 100);
    const scoreColor = score >= 75 ? '#1a9e8f' : score >= 55 ? '#d4a53a' : '#e8622a';

    // Vibe tags
    const queryVibes = new Set((intent?.vibe || []).map((v) => v.toLowerCase()));

    const pct = (val) => Math.round(val * 100);

    const handleBook = (e) => {
        e.stopPropagation();
        onBook(e, p);
    };

    return (
        <div className="card" onClick={() => onClick(p)}>
            <div className="card-header">
                <span className="card-badge">{p.type}</span>
                <div className="card-score-ring">
                    <div className="score-circle" title="Composite match score">
                        <svg width="48" height="48" viewBox="0 0 48 48">
                            <circle cx="24" cy="24" r="20" fill="none" stroke="#f0ede8" strokeWidth="3" />
                            <circle
                                cx="24"
                                cy="24"
                                r="20"
                                fill="none"
                                stroke={scoreColor}
                                strokeWidth="3"
                                strokeDasharray={`${2 * Math.PI * 20}`}
                                strokeDashoffset={`${2 * Math.PI * 20 * (1 - score / 100)}`}
                                strokeLinecap="round"
                                style={{ transition: 'stroke-dashoffset 0.8s ease' }}
                            />
                        </svg>
                        <span style={{ color: scoreColor, fontSize: '12px' }}>{score}%</span>
                    </div>
                </div>
            </div>

            <div className="card-body">
                <div className="card-name">{p.name}</div>
                <div className="card-location">
                    <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" />
                    </svg>
                    {p.neighborhood}, {p.city}, {p.country}
                </div>

                <div className="card-description">{p.description}</div>

                <div className="vibe-tags">
                    {p.vibe_tags.map((v, i) => (
                        <span
                            key={i}
                            className={`vibe-tag ${queryVibes.has(v.toLowerCase()) ? 'matched' : ''}`}
                        >
                            {v}
                        </span>
                    ))}
                </div>

                <div className="match-reason">
                    <span className="match-reason-icon">✦</span>
                    <span className="match-reason-text">{p.match_reason}</span>
                </div>

                {p.review_highlights?.[0] && (
                    <div className="review-highlight">"{p.review_highlights[0]}"</div>
                )}
            </div>

            <div className="card-footer">
                <div>
                    <div className="price-label">from</div>
                    <div className="price-amount">
                        ${p.price_per_night}
                        <small style={{ fontSize: '14px', fontFamily: 'var(--font-body)', fontWeight: 400, color: 'var(--text-3)' }}>
                            /night
                        </small>
                    </div>
                </div>
                <div className="rating-row">
                    <div className="rating-badge">{p.rating}</div>
                    <div className="rating-text">{Number(p.review_count).toLocaleString()} reviews</div>
                </div>
                <button className="book-btn" onClick={handleBook}>
                    Book →
                </button>
            </div>
        </div>
    );
};

export default PropertyCard;
