import React, { useState } from 'react';
import { EDUCATIONAL_CONCEPTS } from '@/lib/educationalContent';
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Lightbulb,
  CheckCircle2,
  Code2,
} from 'lucide-react';

export const ConceptCardsSection: React.FC = () => {
  const [expanded, setExpanded] = useState<boolean>(true);
  const [revealedIds, setRevealedIds] = useState<Record<string, boolean>>({});

  const toggleReveal = (id: string) => {
    setRevealedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <section className="mt-8 rounded-2xl bg-navy-900/90 border border-slate-800 shadow-xl overflow-hidden">
      <div
        onClick={() => setExpanded((v) => !v)}
        className="p-4 bg-navy-800/80 border-b border-slate-800 flex items-center justify-between cursor-pointer hover:bg-navy-800 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <span className="p-2 rounded-xl bg-violet-500/20 text-violet-300">
            <BookOpen className="w-5 h-5" />
          </span>
          <div>
            <h3 className="text-sm md:text-base font-bold text-white">
              Fiches Pédagogiques Masterclass & Passerelles scikit-learn (8 Concepts Clés)
            </h3>
            <p className="text-xs text-slate-400">
              Explications intuitives, défis « Predict before you train » et messages clés à retenir
            </p>
          </div>
        </div>

        <button
          type="button"
          className="px-3 py-1.5 rounded-lg bg-navy-900 border border-slate-700 text-xs font-semibold text-slate-300 flex items-center gap-1"
        >
          {expanded ? (
            <>
              Masquer <ChevronUp className="w-4 h-4" />
            </>
          ) : (
            <>
              Afficher les 7 fiches <ChevronDown className="w-4 h-4" />
            </>
          )}
        </button>
      </div>

      {expanded && (
        <div className="p-5 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {EDUCATIONAL_CONCEPTS.map((concept) => {
            const isRevealed = !!revealedIds[concept.id];
            return (
              <div
                key={concept.id}
                className="p-4 rounded-xl bg-navy-950/90 border border-slate-800/90 hover:border-turquoise/40 transition-all flex flex-col justify-between space-y-3"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-turquoise/15 text-turquoise">
                      {concept.term}
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-white">{concept.titleFr}</h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {concept.beginnerExplanationFr}
                  </p>

                  <div className="p-2.5 rounded-lg bg-navy-900 border border-slate-800 text-[11px] text-violet-300 flex items-start gap-2">
                    <Code2 className="w-4 h-4 shrink-0 mt-0.5 text-violet-400" />
                    <div>
                      <strong className="text-white">Pont scikit-learn :</strong>{' '}
                      {concept.sklearnBridgeFr}
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-400">
                    <strong className="text-turquoise">Manipuler en direct :</strong>{' '}
                    {concept.interactiveTipFr}
                  </div>
                </div>

                {/* Mini Predict before you train challenge */}
                <div className="pt-2 border-t border-slate-800/80 space-y-2">
                  <div className="text-xs font-semibold text-orange-300 flex items-start gap-1.5">
                    <HelpCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>{concept.predictQuestionFr}</span>
                  </div>

                  {!isRevealed ? (
                    <button
                      type="button"
                      onClick={() => toggleReveal(concept.id)}
                      className="text-[11px] font-bold text-turquoise hover:underline"
                    >
                      → Révéler la réponse & le message clé
                    </button>
                  ) : (
                    <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-[11px] text-slate-200 space-y-1">
                      <div className="flex items-start gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-turquoise shrink-0 mt-0.5" />
                        <span>{concept.predictAnswerFr}</span>
                      </div>
                      <div className="text-turquoise font-semibold flex items-center gap-1 pt-1">
                        <Lightbulb className="w-3.5 h-3.5 shrink-0" />
                        <span>À retenir : {concept.takeawayFr}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};
