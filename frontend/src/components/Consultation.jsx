import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import './Consultation.css';

const API = import.meta.env.VITE_API_URL || '';

// ─── Icons ────────────────────────────────────────────────────────────────────

const BotIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="8" r="5" />
    <path d="M20 21a8 8 0 1 0-16 0" />
  </svg>
);

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const CameraIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);

const XIcon = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// ─── Sub-components ───────────────────────────────────────────────────────────

const DiagnosisCard = ({ text, isQuestion }) => {
  if (!text) return null;
  if (isQuestion) {
    return (
      <div className="intake-question">
        <span className="intake-icon">🩺</span>
        <div className="intake-body">
          {text.split('\n').filter(l => l.trim()).map((l, i) => <p key={i}>{l}</p>)}
        </div>
      </div>
    );
  }
  const paragraphs = text.split(/\n{2,}/).filter(p => p.trim());
  return (
    <div className="diagnosis-card">
      <div className="diagnosis-label">
        <span className="diagnosis-icon">🔬</span>
        Clinical Assessment
        <span className="diag-disclaimer">AI · Not a diagnosis</span>
      </div>
      <div className="diagnosis-body">
        {paragraphs.map((para, i) => (
          <p key={i} className="diag-para">{para.trim()}</p>
        ))}
      </div>
    </div>
  );
};

const IngredientRationaleCard = ({ text }) => {
  if (!text) return null;
  return (
    <div className="ingredient-rationale-card">
      <div className="ingredient-rationale-label">
        <span className="ingredient-rationale-icon">⚗️</span>
        Why these ingredients
        <span className="ingredient-rationale-source">from dermatology KB</span>
      </div>
      <p className="ingredient-rationale-body">{text}</p>
    </div>
  );
};

const ProductCard = ({ product }) => (
  <div className="product-card">
    <div className="product-card-header">
      <span className="product-brand">{product.brand}</span>
      {product.format && <span className="product-format">{product.format}</span>}
    </div>
    <h4 className="product-name">{product.name}</h4>
    <p className="product-desc">{product.description}</p>
    {product.key_ingredients?.length > 0 && (
      <div className="product-ingredients">
        {product.key_ingredients.slice(0, 3).map((ing, i) => (
          <span key={i} className="ingredient-tag">{ing}</span>
        ))}
      </div>
    )}
    {product.how_to_use && (
      <p className="product-usage"><strong>How to use:</strong> {product.how_to_use}</p>
    )}
    {product.price_range && <div className="product-price">{product.price_range}</div>}
  </div>
);

const SkinAnalysisCard = ({ analysis }) => {
  if (!analysis || !analysis.clinical_observation) return null;
  return (
    <div className="skin-analysis-card">
      <div className="skin-analysis-label"><span>📸</span> Photo Analysis</div>
      <div className="skin-analysis-row">
        {analysis.skin_type && analysis.skin_type !== 'unknown' && (
          <span className="skin-tag">{analysis.skin_type} skin</span>
        )}
        {analysis.conditions?.map((c, i) => <span key={i} className="skin-tag">{c}</span>)}
      </div>
      <p className="skin-analysis-obs">{analysis.clinical_observation}</p>
    </div>
  );
};

const TrendChart = ({ trends }) => {
  if (!trends) return null;
  const keyMetrics = ['acne_severity', 'redness', 'hair_thinning'];
  let chartData = null;
  let chartTitle = '';

  for (const k of keyMetrics) {
    if (trends[k]?.length >= 2 && trends[k].some(d => d.value > 0)) {
      chartData = trends[k];
      chartTitle = k.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
      break;
    }
  }

  if (!chartData) {
    const firstKey = Object.keys(trends).find(k => trends[k]?.length >= 2 && trends[k].some(d => d.value > 0));
    if (firstKey) {
      chartData = trends[firstKey];
      chartTitle = firstKey.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  }

  if (!chartData) return null;
  const maxVal = Math.max(...chartData.map(d => d.value), 10);

  return (
    <div className="trend-chart-container">
      <div className="trend-title">{chartTitle} Trend</div>
      <div className="trend-bars">
        {chartData.slice(-5).map((d, i) => (
          <div key={i} className="trend-bar-wrapper" title={`Value: ${d.value}`}>
            <div className="trend-bar" style={{ height: `${Math.max((d.value / maxVal) * 100, 5)}%` }} />
            <div className="trend-date">{d.date.slice(5).replace('-', '/')}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ProgressCard = ({ report }) => {
  if (!report) return null;

  const hasComparison = !!report.comparison;
  const hasInsights = report.insight_summary?.length > 0;
  const isFirstUpload = !hasComparison && report.records?.length <= 1;

  if (!hasComparison && !hasInsights && !isFirstUpload) return null;

  return (
    <div className="progress-card">
      <div className="progress-label"><span className="progress-icon">📈</span> Progress Tracking</div>

      {isFirstUpload && (
        <div className="progress-first-upload">
          This is your first recorded upload. Upload again later to track how your skin changes over time.
        </div>
      )}

      {hasComparison && report.comparison.previous_date && (
        <div className="progress-compare-note">
          Compared to your upload on {report.comparison.previous_date}
        </div>
      )}

      {report.lighting_warning && (
        <div className="progress-warning">
          <span>⚠️</span> <strong>Note:</strong> Lighting or angle differs from your last upload — comparisons may be less reliable.
        </div>
      )}

      {hasInsights && (
        <div className="progress-insights">
          {report.insight_summary.map((insight, idx) => (
            <div key={idx} className="progress-insight-item" dangerouslySetInnerHTML={{ __html: insight.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
          ))}
        </div>
      )}

      {report.trends && <TrendChart trends={report.trends} />}

      {report.comparison?.deltas && (
        <div className="progress-metrics-grid">
          {Object.entries(report.comparison.deltas).map(([key, data]) => {
            if (data.neutral) return null;
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return (
              <div key={key} className={`metric-badge ${data.improved ? 'improved' : 'worsened'}`}>
                <span className="metric-name">{label}</span>
                <span className="metric-status">{data.improved ? '↓ Improved' : '↑ Watch'}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

const TypingDots = () => (
  <div className="typing-dots">
    <span /><span /><span />
  </div>
);

const SeverityBadge = ({ concern, severity }) => {
  if (!concern || concern === 'none' || concern === 'unclear_image') return null;
  return (
    <div className="severity-badge">
      <span className={`sev-dot sev-${severity}`} />
      <span>{concern.replace(/_/g, ' ')}</span>
      {severity && <span className="sev-label">{severity}</span>}
    </div>
  );
};

// ─── Loading stage messages ───────────────────────────────────────────────────

const IMAGE_STAGES = [
  'Uploading photo…',
  'Running vision analysis…',
  'Extracting skin observations…',
  'Checking diagnosis…',
  'Searching product database…',
  'Identifying active ingredients from KB…',
  'Generating grounded recommendation…',
];

const TEXT_STAGES = [
  'Thinking…',
  'Assessing concern…',
  'Retrieving context…',
  'Generating response…',
];

// ─── Welcome message ──────────────────────────────────────────────────────────

const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  type: 'text',
  content: "Audito — skin & hair health tracker.\n\nDescribe a concern or upload a photo to begin. Each analysis is logged so you can track how your skin and hair metrics change over time across sessions.",
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Consultation() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [pendingImage, setPendingImage] = useState(null); // { file, previewUrl }
  const [conversationHistory, setConversationHistory] = useState([]);

  const fileRef = useRef(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Loading stage cycling
  useEffect(() => {
    if (!loading) { setLoadingStage(''); return; }
    const stages = pendingImage ? IMAGE_STAGES : TEXT_STAGES;
    let i = 0;
    setLoadingStage(stages[0]);
    const t = setInterval(() => { i = (i + 1) % stages.length; setLoadingStage(stages[i]); }, 3500);
    return () => clearInterval(t);
  }, [loading]); // eslint-disable-line react-hooks/exhaustive-deps

  const addMsg = useCallback((msg) => {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), ...msg }]);
  }, []);

  // ── Select image — don't upload yet, just preview ─────────────────────────

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    if (pendingImage) URL.revokeObjectURL(pendingImage.previewUrl);
    setPendingImage({ file, previewUrl: URL.createObjectURL(file) });
    inputRef.current?.focus();
  };

  const removePendingImage = () => {
    if (pendingImage) URL.revokeObjectURL(pendingImage.previewUrl);
    setPendingImage(null);
  };

  // ── Unified send — handles text-only and image+text ───────────────────────

  const handleSend = async () => {
    const text = input.trim();
    if ((!text && !pendingImage) || loading) return;

    setInput('');
    setLoading(true);

    if (pendingImage) {
      // ── Image path (with optional context text) ──────────────────────────
      const { file, previewUrl } = pendingImage;
      setPendingImage(null);

      addMsg({ role: 'user', type: 'image', content: previewUrl, filename: file.name, caption: text || null });

      const form = new FormData();
      form.append('file', file);
      if (text) form.append('message', text);

      try {
        const { data } = await axios.post(`${API}/api/analyze-image`, form, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 90000,
        });

        addMsg({
          role: 'assistant',
          type: 'response',
          content: data.recommendation || data.diagnosis || 'Analysis complete.',
          intent: data.intent,
          concern: data.identified_concern,
          severity: data.severity,
          diagnosis: data.diagnosis,
          ingredientRationale: data.ingredient_rationale || '',
          recommendation: data.recommendation,
          skinAnalysis: data.skin_analysis,
          progressReport: data.progress_report,
          products: data.show_products ? (data.products || []) : [],
          warnings: data.warnings || [],
          agentPath: data.agent_path || [],
          isIntakeQuestion: !data.recommendation && !!data.diagnosis && (data.agent_path || []).slice(-1)[0] === 'diagnosis',
        });

      } catch (err) {
        const msg = err.code === 'ECONNABORTED'
          ? 'The analysis timed out (server took too long). Please try again — it may be faster on a second attempt once the model is warmed up.'
          : `Could not process the image. ${err.response?.data?.detail || 'Please ensure the photo is clear and well-lit.'}`;
        addMsg({ role: 'assistant', type: 'text', content: msg });
      } finally {
        setLoading(false);
      }

    } else {
      // ── Text-only path ───────────────────────────────────────────────────
      addMsg({ role: 'user', type: 'text', content: text });

      const updatedHistory = [...conversationHistory, { role: 'user', content: text }];

      try {
        const { data } = await axios.post(`${API}/api/chat`, {
          message: text,
          conversation_history: updatedHistory,
        }, { timeout: 90000 });

        const assistantContent = data.recommendation || data.diagnosis || 'I could not generate a response.';

        addMsg({
          role: 'assistant',
          type: 'response',
          content: assistantContent,
          intent: data.intent,
          concern: data.concern,
          severity: data.severity,
          diagnosis: data.diagnosis,
          ingredientRationale: data.ingredient_rationale || '',
          recommendation: data.recommendation,
          products: data.show_products ? (data.products || []) : [],
          warnings: data.warnings || [],
          agentPath: data.agent_path || [],
          isIntakeQuestion: !data.recommendation && !!data.diagnosis && (data.agent_path || []).slice(-1)[0] === 'diagnosis',
        });

        setConversationHistory([...updatedHistory, { role: 'assistant', content: assistantContent }]);

      } catch (err) {
        const msg = err.code === 'ECONNABORTED'
          ? 'Request timed out. Please try again.'
          : `Something went wrong. ${err.response?.data?.detail || 'Please try again.'}`;
        addMsg({ role: 'assistant', type: 'text', content: msg });
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // ─── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">A</div>
          <div>
            <span className="logo-name">Audito</span>
            <span className="logo-sub">Skin &amp; Hair Diagnostic Assistant</span>
          </div>
        </div>
        <span className="header-badge">Database-grounded</span>
      </header>

      {/* Chat area */}
      <main className="chat-area">
        {messages.map((msg) => (
          <div key={msg.id} className={`msg-row ${msg.role}`}>
            {msg.role === 'assistant' && <div className="avatar"><BotIcon /></div>}
            <div className="msg-content">

              {/* Image bubble (user) */}
              {msg.type === 'image' && (
                <div className="image-bubble">
                  <img src={msg.content} alt="uploaded" className="chat-img" />
                  {msg.caption && <div className="text-bubble img-caption-bubble">{msg.caption}</div>}
                  <span className="img-filename">{msg.filename}</span>
                </div>
              )}

              {/* Plain text bubble */}
              {msg.type === 'text' && (
                <div className="text-bubble">
                  {msg.content?.split('\n').map((line, j) => (
                    <span key={j}>{line}{j < msg.content.split('\n').length - 1 && <br />}</span>
                  ))}
                </div>
              )}

              {/* Rich assistant response */}
              {msg.type === 'response' && (
                <>
                  {msg.concern && msg.concern !== 'none' && (
                    <SeverityBadge concern={msg.concern} severity={msg.severity} />
                  )}
                  {msg.skinAnalysis && <SkinAnalysisCard analysis={msg.skinAnalysis} />}
                  {msg.progressReport && <ProgressCard report={msg.progressReport} />}
                  {msg.diagnosis && (
                    <DiagnosisCard
                      text={msg.diagnosis}
                      isQuestion={msg.isIntakeQuestion}
                    />
                  )}
                  {msg.ingredientRationale && !msg.isIntakeQuestion && (
                    <IngredientRationaleCard text={msg.ingredientRationale} />
                  )}
                  {msg.recommendation && (
                    <div className="text-bubble">
                      {msg.recommendation.split('\n').map((line, j) => (
                        <span key={j}>{line}{j < msg.recommendation.split('\n').length - 1 && <br />}</span>
                      ))}
                    </div>
                  )}
                  {msg.warnings?.length > 0 && (
                    <div className="warning-box">
                      {msg.warnings.map((w, j) => <p key={j}>{w}</p>)}
                    </div>
                  )}
                  {msg.products?.length > 0 && (
                    <div className="products-section">
                      <p className="products-label">Recommended from database</p>
                      <div className="products-grid">
                        {msg.products.map((p, j) => <ProductCard key={j} product={p} />)}
                      </div>
                    </div>
                  )}
                  {msg.agentPath?.length > 0 && (
                    <div className="agent-path">{msg.agentPath.join(' → ')}</div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="msg-row assistant">
            <div className="avatar"><BotIcon /></div>
            <div className="msg-content">
              <div className="text-bubble loading-bubble">
                <TypingDots />
                {loadingStage && <span className="loading-stage">{loadingStage}</span>}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Input area */}
      <div className="input-bar">

        {/* Pending image preview strip */}
        {pendingImage && (
          <div className="pending-strip">
            <div className="pending-thumb-wrap">
              <img src={pendingImage.previewUrl} alt="pending upload" className="pending-thumb" />
              <button className="remove-pending" onClick={removePendingImage} title="Remove image">
                <XIcon />
              </button>
            </div>
            <span className="pending-label">Photo attached — add context below or press send</span>
          </div>
        )}

        {/* Input row */}
        <div className="input-row">
          <input
            type="file"
            accept="image/*"
            ref={fileRef}
            style={{ display: 'none' }}
            onChange={handleImageSelect}
          />
          <button
            className={`icon-btn${pendingImage ? ' has-image' : ''}`}
            onClick={() => fileRef.current?.click()}
            disabled={loading}
            title="Attach photo"
          >
            <CameraIcon />
          </button>
          <input
            ref={inputRef}
            className="text-input"
            type="text"
            placeholder={pendingImage ? 'Add context about the photo (optional)…' : 'Describe your skin or hair concern…'}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={(!input.trim() && !pendingImage) || loading}
            title="Send"
          >
            <SendIcon />
          </button>
        </div>
      </div>

      <p className="disclaimer">
        Audito provides informational guidance only — not medical advice. Consult a dermatologist for clinical diagnosis.
      </p>
    </div>
  );
}
