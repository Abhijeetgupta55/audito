import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sparkles, Activity, ShieldCheck, Camera } from 'lucide-react';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <nav className="navbar glass-panel">
        <div className="logo">
          <span className="text-gradient">Clinikally</span>
          <span className="badge">AI Lab</span>
        </div>
        <div className="nav-links">
          <a href="#about">Technology</a>
          <a href="#accuracy">Accuracy</a>
        </div>
      </nav>

      <main className="hero-section">
        <motion.div 
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="badge-wrapper">
            <span className="beta-badge">
              <Sparkles size={14} className="pulse-glow" /> 
              Next-Gen Clinical AI
            </span>
          </div>
          
          <h1 className="hero-title">
            The Future of <br />
            <span className="text-gradient">Skin & Hair Health</span>
          </h1>
          
          <p className="hero-subtitle">
            Experience our advanced AI Agent. Built with multi-agent architecture and RAG for 100% factual analysis. A real-time, clinical-grade consultation from your device.
          </p>

          <div className="cta-group">
            <button 
              className="btn btn-primary btn-large pulse-glow"
              onClick={() => navigate('/consultation')}
            >
              <Camera size={20} />
              Start Live AI Consultation
            </button>
            <button className="btn btn-secondary btn-large">
              View Case Studies
            </button>
          </div>

          <div className="features-grid">
            <FeatureCard 
              icon={<Activity color="var(--accent-primary)" size={24} />}
              title="Real-Time Analysis"
              desc="GPT-4V powered vision tracking for skin concerns."
            />
            <FeatureCard 
              icon={<ShieldCheck color="var(--success)" size={24} />}
              title="Factual & Grounded"
              desc="No hallucinations. Backed by 15k+ products via Pinecone."
            />
            <FeatureCard 
              icon={<Sparkles color="var(--accent-secondary)" size={24} />}
              title="Multi-Agent Pipeline"
              desc="Triage, Vision, and Safety agents working in harmony."
            />
          </div>
        </motion.div>

        <motion.div 
          className="hero-visual animate-float"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <div className="visual-core glass-panel">
            <div className="scanner-line"></div>
            <img src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?q=80&w=600&auto=format&fit=crop" alt="Skin mapping interface" className="demo-img" />
            <div className="overlay-ui">
              <div className="ui-badge success">Analysis Complete</div>
              <div className="ui-badge info">Hydration: Optimal</div>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
};

const FeatureCard = ({ icon, title, desc }) => (
  <div className="feature-card glass-panel">
    <div className="feature-icon">{icon}</div>
    <h3>{title}</h3>
    <p>{desc}</p>
  </div>
);

export default Home;
