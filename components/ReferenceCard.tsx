
import React from 'react';
import type { ReferenceItem } from '@/types';
import { Eye, ArrowLeft, Clock, Database, Terminal } from 'lucide-react';
import CodeEditor from 'react-simple-code-editor';
import Prism from 'prismjs';

interface ReferenceCardProps {
  item: ReferenceItem;
  isFlipped: boolean;
  onFlip: () => void;
}

const ReferenceCard: React.FC<ReferenceCardProps> = ({ item, isFlipped, onFlip }) => {

  const highlight = (code: string) => {
    const grammer = Prism.languages.python || Prism.languages.javascript || Prism.languages.clike;
    return Prism.highlight(code, grammer, 'python');
  };

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case 'Data Structure': return 'bg-purple-200';
      case 'Algorithm': return 'bg-blue-200';
      case 'Concept': return 'bg-pink-200';
      case 'System Design': return 'bg-orange-200';
      default: return 'bg-gray-200';
    }
  };

  return (
    <div className="w-full h-96 perspective-1000 group cursor-pointer">
      <div
        className={`relative w-full h-full transition-transform duration-500 transform-style-3d ${isFlipped ? 'rotate-y-180' : ''
          }`}
      >

        {/* FRONT */}
        <div className="absolute inset-0 backface-hidden bg-white border-2 border-black shadow-retro hover:shadow-retro-lg hover:-translate-y-1 transition-all flex flex-col">
          <div className="p-6 flex flex-col h-full">
            <div className="flex justify-between items-start mb-4">
              <span className={`inline-flex items-center px-2 py-1 text-xs font-mono font-bold border border-black ${getCategoryColor(item.category)}`}>
                {item.category.toUpperCase()}
              </span>
            </div>

            <h2 className="text-xl font-bold font-mono text-black mb-3 uppercase">{item.title}</h2>
            <p className="text-gray-600 text-sm mb-6 flex-1 font-mono leading-relaxed">{item.summary}</p>

            <div className="space-y-2 mb-6 border-t-2 border-gray-100 pt-4">
              <div className="flex items-center text-xs font-mono text-black">
                <Clock size={14} className="mr-2" />
                <span>TIME: {item.complexity.time}</span>
              </div>
              <div className="flex items-center text-xs font-mono text-black">
                <Database size={14} className="mr-2" />
                <span>SPACE: {item.complexity.space}</span>
              </div>
            </div>

            <button
              onClick={(e) => { e.stopPropagation(); onFlip(); }}
              className="w-full flex items-center justify-center gap-2 text-black bg-white border-2 border-black font-mono font-bold text-xs py-2 hover:bg-black hover:text-white transition-colors shadow-retro-sm"
            >
              <Eye size={14} /> VIEW_SOURCE
            </button>
          </div>
        </div>

        {/* BACK */}
        <div className="absolute inset-0 backface-hidden rotate-y-180 bg-black border-2 border-black shadow-retro flex flex-col">
          <div className="p-3 border-b border-gray-800 flex justify-between items-center bg-gray-900">
            <span className="font-bold text-white text-xs font-mono flex items-center gap-2"><Terminal size={14} /> IMPLEMENTATION.PY</span>
            <button
              onClick={(e) => { e.stopPropagation(); onFlip(); }}
              className="text-xs flex items-center gap-1 text-black bg-white px-2 py-1 border border-gray-500 font-bold uppercase"
            >
              <ArrowLeft size={12} /> Back
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-0 bg-black">
            <div className="p-4 text-xs text-gray-400 border-b border-gray-800 bg-gray-900/50 whitespace-pre-wrap font-mono">
              # {item.description.replace(/\n/g, '\n# ')}
            </div>
            <div className="p-4">
              <CodeEditor
                value={item.implementation}
                onValueChange={() => { }}
                highlight={highlight}
                padding={0}
                readOnly
                className="font-mono text-xs text-green-400"
                style={{
                  fontFamily: '"Fira Code", monospace',
                  fontSize: 12,
                }}
                textareaClassName="focus:outline-none"
              />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ReferenceCard;