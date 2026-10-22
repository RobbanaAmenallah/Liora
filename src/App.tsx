import React, { useState } from 'react';
import { ExploreNeuronTab } from '@/components/tabs/ExploreNeuronTab';
import { TrainNetworkTab } from '@/components/tabs/TrainNetworkTab';
import { ChallengeTab } from '@/components/tabs/ChallengeTab';
import { ConceptCardsSection } from '@/components/educational/ConceptCardsSection';
import { GUIDED_EXPERIMENTS } from '@/lib/educationalContent';
import {
  BrainCircuit,
  Cpu,
  Network,
  Trophy,
  Presentation,
  PlayCircle,
  RotateCcw,
  Mail,
  Linkedin,
  Globe,
  Github,
} from 'lucide-react';

type ActiveTab = 'neuron' | 'train' | 'challenge';

export function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('neuron');
  const [presentationMode, setPresentationMode] = useState<boolean>(false);
  const [activeExperimentId, setActiveExperimentId] = useState<
    'A' | 'B' | 'C' | 'D' | null
  >(null);
  const [resetKey, setResetKey] = useState<number>(0);

  const handleStartGuidedDemo = () => {
    setActiveTab('train');
    setActiveExperimentId('A');
  };

  const handleResetDemoDefaults = () => {
    setActiveExperimentId(null);
    setActiveTab('neuron');
    setResetKey((k) => k + 1);
  };

  const currentExp = GUIDED_EXPERIMENTS.find((e) => e.id === activeExperimentId);

  return (
    <div className="min-h-screen bg-[#0D1830] text-slate-100 flex flex-col">
      {/* Top Navigation & Instructor Masterclass Header */}
      <header className="sticky top-0 z-40 bg-navy-950/90 backdrop-blur-md border-b border-slate-800/90 px-4 lg:px-8 py-3">
        <div className="max-w-[1800px] mx-auto flex flex-wrap items-center justify-between gap-4">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-turquoise to-violet-500 p-0.5 shadow-[0_0_20px_rgba(89,217,197,0.3)]">
              <div className="w-full h-full bg-navy-950 rounded-[14px] flex items-center justify-center">
                <BrainCircuit className="w-5 h-5 text-turquoise" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base md:text-lg font-extrabold tracking-tight text-white">
                  Neural Network Lab
                </h1>
              </div>
              <p className="text-xs text-slate-400 hidden md:block">
                Du Machine Learning (scikit-learn) aux Réseaux de Neurones Profonds (TensorFlow.js en temps réel)
              </p>
            </div>
          </div>

          {/* 3 Main Tabs Selector */}
          <nav className="flex items-center gap-1.5 p-1 rounded-2xl bg-navy-900 border border-slate-800">
            <button
              type="button"
              onClick={() => setActiveTab('neuron')}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all ${
                activeTab === 'neuron'
                  ? 'bg-turquoise text-navy-950 shadow-md'
                  : 'text-slate-300 hover:text-white hover:bg-navy-800'
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>1. Explore the Neuron</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('train')}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all ${
                activeTab === 'train'
                  ? 'bg-turquoise text-navy-950 shadow-md'
                  : 'text-slate-300 hover:text-white hover:bg-navy-800'
              }`}
            >
              <Network className="w-4 h-4" />
              <span>2. Train a Neural Network</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('challenge')}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all ${
                activeTab === 'challenge'
                  ? 'bg-turquoise text-navy-950 shadow-md'
                  : 'text-slate-300 hover:text-white hover:bg-navy-800'
              }`}
            >
              <Trophy className="w-4 h-4" />
              <span>3. Neural Network Challenge</span>
            </button>
          </nav>

          {/* Instructor Actions: Démarrer la démonstration & Presentation Mode */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleStartGuidedDemo}
              className="px-3.5 py-2 rounded-xl bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 text-white font-bold text-xs flex items-center gap-1.5 shadow-lg transition-all"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Démarrer la démonstration</span>
            </button>

            <button
              type="button"
              onClick={() => setPresentationMode((v) => !v)}
              className={`px-3.5 py-2 rounded-xl text-xs font-bold flex items-center gap-1.5 border transition-all ${
                presentationMode
                  ? 'bg-orange-500 text-navy-950 border-orange-400 shadow-[0_0_15px_rgba(251,146,60,0.4)]'
                  : 'bg-navy-900 text-slate-200 border-slate-700 hover:border-turquoise/50'
              }`}
              title="Optimise l'affichage pour un partage d'écran Zoom 1920×1080"
            >
              <Presentation className="w-4 h-4" />
              <span>Presentation Mode</span>
            </button>
          </div>
        </div>

        {/* Presentation Mode Quick Bar (visible when Presentation Mode is enabled) */}
        {presentationMode && (
          <div className="max-w-[1800px] mx-auto mt-3 pt-2.5 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-orange-500/20 text-orange-300 font-bold uppercase text-[10px]">
                Mode Zoom 1080p Actif
              </span>
              <span className="text-slate-200 font-semibold">
                Objectif pédagogique actuel :{' '}
                <strong className="text-turquoise">
                  {currentExp
                    ? `${currentExp.title} — ${currentExp.subtitle}`
                    : activeTab === 'neuron'
                    ? 'Comprendre le calcul z = w₁x₁ + w₂x₂ + b et le rôle de la non-linéarité φ(z)'
                    : activeTab === 'train'
                    ? 'Visualiser comment l’architecture et les couches cachées sculptent la Decision Boundary'
                    : 'Valider les acquis par le diagnostic expérimental (Missions 1 à 3)'}
                </strong>
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-slate-400 font-medium">Accès rapide Expériences :</span>
              {(['A', 'B', 'C', 'D'] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => {
                    setActiveTab('train');
                    setActiveExperimentId(id);
                  }}
                  className={`px-2.5 py-1 rounded-lg font-bold transition-all ${
                    activeExperimentId === id
                      ? 'bg-turquoise text-navy-950'
                      : 'bg-navy-800 text-slate-300 hover:text-white'
                  }`}
                >
                  Exp. {id}
                </button>
              ))}

              <button
                type="button"
                onClick={handleResetDemoDefaults}
                className="ml-2 px-2.5 py-1 rounded-lg bg-navy-800 hover:bg-navy-700 border border-slate-700 text-slate-200 flex items-center gap-1"
              >
                <RotateCcw className="w-3.5 h-3.5 text-turquoise" />
                Reset Demo Defaults
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1800px] w-full mx-auto px-4 lg:px-8 py-6">
        {activeTab === 'neuron' && (
          <ExploreNeuronTab presentationMode={presentationMode} />
        )}

        {activeTab === 'train' && (
          <TrainNetworkTab
            key={resetKey}
            presentationMode={presentationMode}
            activeExperimentId={activeExperimentId}
            onSelectExperiment={setActiveExperimentId}
          />
        )}

        {activeTab === 'challenge' && <ChallengeTab />}

        {/* Educational Concept Reference Cards (hidden in Presentation Mode to keep screen uncluttered) */}
        {!presentationMode && <ConceptCardsSection />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-navy-950/95 py-4 px-4 lg:px-8 text-xs text-slate-400">
        <div className="max-w-[1800px] mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 text-center md:text-left">
            <strong className="text-white font-bold text-sm">
              Robbana Amenallah — Liora
            </strong>
            <span className="text-slate-600 hidden sm:inline">•</span>
            <span className="text-slate-400">
              Neural Network Lab — Masterclass Deep Learning Interactive
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <a
              href="mailto:amenallah.robbana@dauphine.eu"
              className="px-2.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 border border-slate-800 hover:border-turquoise/50 text-slate-200 hover:text-turquoise transition-all inline-flex items-center gap-1.5"
            >
              <Mail className="w-3.5 h-3.5 text-turquoise" />
              <span>amenallah.robbana@dauphine.eu</span>
            </a>

            <a
              href="https://www.linkedin.com/in/robbana-amenallah/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-2.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 border border-slate-800 hover:border-sky-400/50 text-slate-200 hover:text-sky-300 transition-all inline-flex items-center gap-1.5"
            >
              <Linkedin className="w-3.5 h-3.5 text-sky-400" />
              <span>LinkedIn</span>
            </a>

            <a
              href="https://robbanaamenallah.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-2.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 border border-slate-800 hover:border-violet-400/50 text-slate-200 hover:text-violet-300 transition-all inline-flex items-center gap-1.5"
            >
              <Globe className="w-3.5 h-3.5 text-violet-400" />
              <span>Portfolio</span>
            </a>

            <a
              href="https://github.com/RobbanaAmenallah"
              target="_blank"
              rel="noopener noreferrer"
              className="px-2.5 py-1.5 rounded-lg bg-navy-900 hover:bg-navy-800 border border-slate-800 hover:border-orange-400/50 text-slate-200 hover:text-orange-300 transition-all inline-flex items-center gap-1.5"
            >
              <Github className="w-3.5 h-3.5 text-orange-400" />
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
