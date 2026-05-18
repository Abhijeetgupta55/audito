import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './DiagnosticDeck.css';

// ── Card builder ───────────────────────────────────────────────────────────────

export function buildCards(msg) {
  // Intake question — single card, no deck nav
  if (msg.isIntakeQuestion && msg.diagnosis) {
    return [{ type: 'intake', text: msg.diagnosis }];
  }

  // Pure conversational — return empty so Consultation renders text bubble
  const hasStructure = !!(
    msg.skinAnalysis?.clinical_observation ||
    msg.diagnosisData?.diagnosis_summary?.length ||
    msg.actives?.length ||
    msg.ingredientRationale ||
    msg.products?.length ||
    msg.stage2Pending ||
    msg.stage2Loading ||
    msg.progressReport ||
    msg.diagnosis
  );
  if (!hasStructure) return [];

  const cards = [];

  // 1 — Photo analysis
  if (msg.skinAnalysis?.clinical_observation) {
    cards.push({ type: 'photo', data: msg.skinAnalysis, lowConfidence: !!(msg.skinAnalysis?.low_confidence || msg.lowConfidence) });
  }

  // 2 — Progress tracking
  const pr = msg.progressReport;
  if (pr) {
    const hasContent = pr.comparison || pr.insight_summary?.length > 0 || pr.records?.length <= 1;
    if (hasContent) cards.push({ type: 'progress', report: pr });
  }

  // 3 — Clinical summary
  if (msg.diagnosisData?.diagnosis_summary?.length) {
    cards.push({
      type: 'summary',
      concerns: msg.diagnosisData.concerns || [],
      severity: msg.diagnosisData.severity || msg.severity,
      bullets: msg.diagnosisData.diagnosis_summary,
    });
  } else if (msg.diagnosis) {
    const lines = msg.diagnosis.split('\n').filter(l => l.trim()).map(l => l.replace(/^[•\-*]\s*/, ''));
    if (lines.length) {
      cards.push({ type: 'summary', concerns: [], severity: msg.severity, bullets: lines });
    }
  }

  // 4 — Recommended actives (with placeholder if unavailable)
  if (msg.actives?.length) {
    cards.push({ type: 'actives', actives: msg.actives });
  } else if (msg.ingredientRationale) {
    const parsed = msg.ingredientRationale.split('\n').filter(l => l.trim()).map(l => {
      const clean = l.replace(/^[•\-*]\s*/, '');
      const parts = clean.split(' — ');
      return parts.length >= 3
        ? { name: parts[0], mechanism: parts[1], target_concern: parts.slice(2).join(' — ') }
        : null;
    }).filter(Boolean);
    if (parsed.length) {
      cards.push({ type: 'actives', actives: parsed });
    } else {
      cards.push({ type: 'actives_pending' });
    }
  } else if (msg.stage2Pending) {
    // Image analysis ran and concern confirmed, but actives not yet generated
    cards.push({ type: 'actives_pending' });
  }

  // 5 — Warning / caution
  const isSevere = msg.severity === 'severe';
  const needsDoctor = msg.diagnosisData?.requires_doctor;
  const hasWarning = isSevere || needsDoctor || msg.warnings?.length;
  if (hasWarning) {
    let text = '';
    if (isSevere) text = 'Severe presentation — dermatologist consultation recommended before starting any treatment.';
    else if (needsDoctor) text = 'Professional consultation is recommended for this concern.';
    else if (msg.warnings?.length) text = msg.warnings[0].replace(/^⚠️?\s*/i, '');
    if (text) cards.push({ type: 'warning', text, severe: isSevere });
  }

  // 6 — Products (loading / CTA / loaded)
  if (msg.stage2Loading) {
    cards.push({ type: 'products_loading' });
  } else if (msg.stage2Pending) {
    cards.push({ type: 'products_cta', stage2Data: msg.stage2Data });
  } else if (msg.products?.length) {
    cards.push({ type: 'products', products: msg.products });
  }

  return cards;
}

// ── Spring config ──────────────────────────────────────────────────────────────

const SPRING = { type: 'spring', stiffness: 420, damping: 36, mass: 0.75 };

// ── Deck navigation ────────────────────────────────────────────────────────────

const DeckNav = ({ current, total, onNext, onPrev }) => {
  if (total <= 1) return null;
  return (
    <div className="dn-nav">
      <button
        className="dn-nav-btn"
        onClick={onPrev}
        disabled={current === 0}
        aria-label="Previous card"
      >
        ←
      </button>
      <div className="dn-dots" aria-label={`Card ${current + 1} of ${total}`}>
        {Array.from({ length: total }, (_, i) => (
          <span key={i} className={`dn-dot${i === current ? ' dn-dot-on' : ''}`} />
        ))}
      </div>
      <button
        className="dn-nav-btn"
        onClick={onNext}
        disabled={current >= total - 1}
        aria-label="Next card"
      >
        →
      </button>
    </div>
  );
};

// ── Trend chart ────────────────────────────────────────────────────────────────

const MiniTrend = ({ trends }) => {
  const keys = ['acne_severity', 'redness', 'hair_thinning'];
  let data = null, title = '';
  for (const k of keys) {
    if (trends[k]?.length >= 2 && trends[k].some(d => d.value > 0)) {
      data = trends[k]; title = k.replace(/_/g, ' '); break;
    }
  }
  if (!data) {
    const fk = Object.keys(trends).find(k => trends[k]?.length >= 2 && trends[k].some(d => d.value > 0));
    if (fk) { data = trends[fk]; title = fk.replace(/_/g, ' '); }
  }
  if (!data) return null;
  const max = Math.max(...data.map(d => d.value), 10);
  return (
    <div className="dn-trend">
      <div className="dn-trend-label">{title}</div>
      <div className="dn-trend-bars">
        {data.slice(-5).map((d, i) => (
          <div key={i} className="dn-bar-col">
            <div className="dn-bar" style={{ height: `${Math.max((d.value / max) * 100, 6)}%` }} />
            <div className="dn-bar-date">{d.date.slice(5).replace('-', '/')}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Individual card components ─────────────────────────────────────────────────

const PhotoCard = ({ card, nav }) => (
  <div className="dc dc-photo">
    <div className="dc-head">
      <span className="dc-label">📸 Photo Analysis</span>
      {card.lowConfidence && <span className="dc-badge-warn">Limited quality</span>}
    </div>
    <div className="dc-body">
      {card.lowConfidence && card.data.clarity_feedback && (
        <p className="dc-inline-warn">⚠ {card.data.clarity_feedback}</p>
      )}
      {(card.data.skin_type && card.data.skin_type !== 'unknown' || card.data.conditions?.length > 0) && (
        <div className="dc-tag-row">
          {card.data.skin_type && card.data.skin_type !== 'unknown' && (
            <span className="dc-tag dc-tag-violet">{card.data.skin_type} skin</span>
          )}
          {card.data.conditions?.map((c, i) => (
            <span key={i} className="dc-tag dc-tag-violet">{c}</span>
          ))}
        </div>
      )}
      <p className="dc-obs">{card.data.clinical_observation}</p>
    </div>
    {nav}
  </div>
);

const ProgressCard = ({ card, nav }) => {
  const pr = card.report;
  const isFirst = !pr.comparison && pr.records?.length <= 1;
  return (
    <div className="dc dc-progress">
      <div className="dc-head">
        <span className="dc-label">📈 Progress</span>
        {pr.comparison?.previous_date && (
          <span className="dc-badge">vs {pr.comparison.previous_date}</span>
        )}
      </div>
      <div className="dc-body">
        {isFirst && (
          <p className="dc-body-text">First upload recorded. Scan again later to track changes over time.</p>
        )}
        {pr.lighting_warning && (
          <p className="dc-inline-warn">⚠ Lighting or angle differs — comparisons may be less reliable.</p>
        )}
        {pr.insight_summary?.length > 0 && (
          <ul className="dc-list">
            {pr.insight_summary.slice(0, 3).map((s, i) => (
              <li key={i} dangerouslySetInnerHTML={{ __html: s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
            ))}
          </ul>
        )}
        {pr.trends && <MiniTrend trends={pr.trends} />}
        {pr.comparison?.deltas && (
          <div className="dc-metric-row">
            {Object.entries(pr.comparison.deltas).filter(([, d]) => !d.neutral).map(([key, data]) => (
              <span key={key} className={`dc-metric ${data.improved ? 'dc-metric-up' : 'dc-metric-dn'}`}>
                {key.replace(/_/g, ' ')} {data.improved ? '↓' : '↑'}
              </span>
            ))}
          </div>
        )}
      </div>
      {nav}
    </div>
  );
};

const SummaryCard = ({ card, nav }) => (
  <div className="dc dc-summary">
    <div className="dc-head">
      <span className="dc-label">🔬 Clinical Summary</span>
      <div className="dc-head-right">
        {card.severity && card.severity !== 'unknown' && (
          <span className={`dc-sev dc-sev-${card.severity}`}>{card.severity}</span>
        )}
        <span className="dc-disclaimer">AI · Not a diagnosis</span>
      </div>
    </div>
    <div className="dc-body">
      {card.concerns?.length > 0 && (
        <div className="dc-tag-row">
          {card.concerns.map((c, i) => (
            <span key={i} className="dc-tag dc-tag-blue">{c}</span>
          ))}
        </div>
      )}
      <ul className="dc-bullets">
        {card.bullets.map((b, i) => <li key={i}>{b}</li>)}
      </ul>
    </div>
    {nav}
  </div>
);

const ActivesCard = ({ card, nav }) => (
  <div className="dc dc-actives">
    <div className="dc-head dc-head-dark">
      <span className="dc-label-light">⚗️ Recommended Actives</span>
      <span className="dc-badge-faint">KB-grounded</span>
    </div>
    <div className="dc-body">
      <div className="dc-actives-list">
        {card.actives.map((a, i) => (
          <div key={i} className="dc-active-row">
            <span className="dc-active-name">{a.name}</span>
            <div className="dc-active-detail">
              <span className="dc-active-arrow">→</span>
              <span className="dc-active-mech">{a.mechanism}</span>
              <span className="dc-active-pill">{a.target_concern}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
    {nav}
  </div>
);

const ActivesPendingCard = ({ nav }) => (
  <div className="dc dc-actives">
    <div className="dc-head dc-head-dark">
      <span className="dc-label-light">⚗️ Recommended Actives</span>
    </div>
    <div className="dc-body dc-actives-pending-body">
      <div className="dc-actives-pending-row">
        <div className="dc-spinner" />
        <span className="dc-actives-pending-text">Ingredient analysis still refining…</span>
      </div>
    </div>
    {nav}
  </div>
);

const WarningCard = ({ card, nav }) => (
  <div className={`dc dc-warning${card.severe ? ' dc-warning-severe' : ''}`}>
    <div className="dc-head">
      <span className="dc-label">{card.severe ? '🚨 Important' : '⚠ Note'}</span>
    </div>
    <div className="dc-body">
      <p className="dc-warning-text">{card.text}</p>
    </div>
    {nav}
  </div>
);

const ProductsCTACard = ({ card, nav, onCTA }) => (
  <div className="dc dc-cta">
    <div className="dc-head">
      <span className="dc-label">💊 Products</span>
    </div>
    <div className="dc-body dc-cta-body">
      <p className="dc-cta-headline">Ready for product recommendations?</p>
      <p className="dc-cta-sub">Matched to your findings from our dermatology database.</p>
      <button className="dc-cta-btn" onClick={() => onCTA(card.stage2Data)}>
        Get Product Recommendations
      </button>
    </div>
    {nav}
  </div>
);

const ProductsLoadingCard = ({ nav }) => (
  <div className="dc dc-cta">
    <div className="dc-head">
      <span className="dc-label">💊 Products</span>
    </div>
    <div className="dc-body dc-cta-body">
      <div className="dc-loader-row">
        <div className="dc-spinner" />
        <span className="dc-loader-text">Searching product database…</span>
      </div>
    </div>
    {nav}
  </div>
);

const ProductsCard = ({ card, nav }) => (
  <div className="dc dc-products">
    <div className="dc-head">
      <span className="dc-label">💊 Matched Products</span>
      <span className="dc-badge">database</span>
    </div>
    <div className="dc-body dc-products-body">
      {card.products.map((p, i) => (
        <div key={i} className={`dc-product${i < card.products.length - 1 ? ' dc-product-sep' : ''}`}>
          <div className="dc-product-row1">
            <span className="dc-product-brand">{p.brand}</span>
            {p.format && <span className="dc-product-format">{p.format}</span>}
            {p.price_range && <span className="dc-product-price">{p.price_range}</span>}
          </div>
          <div className="dc-product-name">{p.name}</div>
          <p className="dc-product-desc">{p.description}</p>
          {p.key_ingredients?.length > 0 && (
            <div className="dc-tag-row">
              {p.key_ingredients.slice(0, 3).map((ing, j) => (
                <span key={j} className="dc-tag dc-tag-green">{ing}</span>
              ))}
            </div>
          )}
          {p.how_to_use && (
            <p className="dc-product-use"><strong>Use:</strong> {p.how_to_use}</p>
          )}
        </div>
      ))}
    </div>
    {nav}
  </div>
);

const IntakeCard = ({ card }) => (
  <div className="dc dc-intake">
    <div className="dc-head">
      <span className="dc-label">🩺 A few questions</span>
    </div>
    <div className="dc-body">
      {card.text.split('\n').filter(l => l.trim()).map((line, i) => (
        <p key={i} className="dc-intake-line">{line}</p>
      ))}
    </div>
  </div>
);

// ── Main DiagnosticDeck component ──────────────────────────────────────────────

export default function DiagnosticDeck({ msg, onGetProducts }) {
  const [activeIdx, setActiveIdx] = useState(0);
  const touchStartX = useRef(null);
  const cards = buildCards(msg);

  // Clamp when cards array changes (e.g. products loaded)
  useEffect(() => {
    if (cards.length > 0 && activeIdx >= cards.length) {
      setActiveIdx(cards.length - 1);
    }
  }, [cards.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCTA = useCallback((stage2Data) => {
    onGetProducts(msg.id, stage2Data);
  }, [msg.id, onGetProducts]);

  const goNext = useCallback(() => setActiveIdx(i => Math.min(i + 1, cards.length - 1)), [cards.length]);
  const goPrev = useCallback(() => setActiveIdx(i => Math.max(i - 1, 0)), []);

  const handleTouchStart = (e) => { touchStartX.current = e.touches[0].clientX; };
  const handleTouchEnd = (e) => {
    if (touchStartX.current === null) return;
    const dx = e.changedTouches[0].clientX - touchStartX.current;
    if (dx < -45) goNext();
    else if (dx > 45) goPrev();
    touchStartX.current = null;
  };

  if (!cards.length) return null;

  const card = cards[activeIdx];
  const total = cards.length;
  const isSingle = total === 1;

  const nav = (
    <DeckNav current={activeIdx} total={total} onNext={goNext} onPrev={goPrev} />
  );

  const renderCard = () => {
    switch (card.type) {
      case 'photo':            return <PhotoCard card={card} nav={nav} />;
      case 'progress':         return <ProgressCard card={card} nav={nav} />;
      case 'summary':          return <SummaryCard card={card} nav={nav} />;
      case 'actives':          return <ActivesCard card={card} nav={nav} />;
      case 'actives_pending':  return <ActivesPendingCard nav={nav} />;
      case 'warning':          return <WarningCard card={card} nav={nav} />;
      case 'products_cta':     return <ProductsCTACard card={card} nav={nav} onCTA={handleCTA} />;
      case 'products_loading': return <ProductsLoadingCard nav={nav} />;
      case 'products':         return <ProductsCard card={card} nav={nav} />;
      case 'intake':           return <IntakeCard card={card} />;
      default:                 return null;
    }
  };

  return (
    <div
      className={`diagnostic-deck${isSingle ? ' deck-single' : ''}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* Peek strips — create depth illusion */}
      {!isSingle && activeIdx + 2 < total && <div className="deck-peek deck-peek-2" />}
      {!isSingle && activeIdx + 1 < total && <div className="deck-peek deck-peek-1" />}

      <AnimatePresence mode="wait">
        <motion.div
          key={activeIdx}
          className="deck-card-wrap"
          initial={{ opacity: 0, y: 10, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.985 }}
          transition={SPRING}
        >
          {renderCard()}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
