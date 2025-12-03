
import React from 'react';
import type { AnalysisResult } from '@/types';
import { CheckCircle, XCircle, Clock, Database, Lightbulb, Zap } from 'lucide-react';

interface AnalysisResultViewProps {
  result: AnalysisResult;
}

const AnalysisResultView: React.FC<AnalysisResultViewProps> = ({ result }) => {
  return (
    <div className="bg-white border-2 border-black shadow-retro p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-8 pb-4 border-b-2 border-gray-100">
        <div className="flex items-center gap-4">
          <div className={`p-2 border-2 border-black ${result.isCorrect ? 'bg-green-400' : 'bg-red-500'}`}>
             {result.isCorrect ? <CheckCircle size={32} className="text-black" /> : <XCircle size={32} className="text-white" />}
          </div>
          <div>
            <h2 className="text-xl font-black font-mono uppercase text-black">
                {result.isCorrect ? "Success" : "Failed"}
            </h2>
            <span className="text-xs font-mono text-gray-500 uppercase">Status Code: {result.isCorrect ? '200 OK' : '400 BAD REQUEST'}</span>
          </div>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-xs text-gray-500 font-mono font-bold uppercase tracking-wider mb-1">Efficiency Score</span>
          <div className="relative">
             <span className="text-4xl font-black font-mono text-black italic">{result.grade}</span>
             <span className="absolute top-0 -right-4 text-xs font-bold text-gray-400">/100</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-[#f0f0f0] p-4 border-2 border-black shadow-retro-sm">
          <div className="flex items-center gap-2 text-black font-mono font-bold text-sm mb-3 border-b-2 border-black pb-1 inline-block">
            <Clock size={16} /> TIME_COMPLEXITY
          </div>
          <p className="text-sm text-gray-800 font-mono leading-relaxed">{result.timeComplexityFeedback}</p>
        </div>
        <div className="bg-[#f0f0f0] p-4 border-2 border-black shadow-retro-sm">
          <div className="flex items-center gap-2 text-black font-mono font-bold text-sm mb-3 border-b-2 border-black pb-1 inline-block">
            <Database size={16} /> SPACE_COMPLEXITY
          </div>
          <p className="text-sm text-gray-800 font-mono leading-relaxed">{result.spaceComplexityFeedback}</p>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <div className="flex items-center gap-2 text-black font-bold font-mono text-sm mb-2 uppercase">
            <Zap size={16} className="text-yellow-500 fill-current" /> Code Quality Audit
          </div>
          <p className="text-sm text-gray-700 border-l-4 border-yellow-400 bg-yellow-50 p-4 font-mono">
            {result.codeQualityFeedback}
          </p>
        </div>
        
        <div>
          <div className="flex items-center gap-2 text-black font-bold font-mono text-sm mb-2 uppercase">
            <Lightbulb size={16} className="text-blue-500 fill-current" /> Optimization Vector
          </div>
          <p className="text-sm text-gray-700 border-l-4 border-blue-400 bg-blue-50 p-4 font-mono">
            {result.suggestions}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResultView;
