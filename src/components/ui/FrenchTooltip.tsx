import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

interface FrenchTooltipProps {
  term: string;
  explanation: string;
  sklearnNote?: string;
}

export const FrenchTooltip: React.FC<FrenchTooltipProps> = ({
  term,
  explanation,
  sklearnNote,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex items-center ml-1.5 align-middle"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-label={`Explication pour ${term}`}
        className="text-slate-400 hover:text-turquoise transition-colors focus:outline-none"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {open && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 rounded-xl bg-navy-950/95 border border-turquoise/40 shadow-2xl z-50 text-left pointer-events-none backdrop-blur-md">
          <div className="text-xs font-bold text-turquoise mb-1 flex items-center justify-between">
            <span>{term}</span>
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-turquoise/10 text-turquoise">
              Glossaire
            </span>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed">{explanation}</p>
          {sklearnNote && (
            <div className="mt-2 pt-1.5 border-t border-slate-800 text-[11px] text-violet-400">
              <span className="font-semibold">Repère scikit-learn :</span> {sklearnNote}
            </div>
          )}
        </div>
      )}
    </span>
  );
};
