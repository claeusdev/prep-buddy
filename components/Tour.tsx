import React, { useState, useLayoutEffect } from 'react';
import { X, ChevronRight } from 'lucide-react';

interface TourStep {
  targetId: string;
  title: string;
  content: string;
}

interface TourProps {
  steps: TourStep[];
  onComplete: () => void;
  onSkip: () => void;
}

const Tour: React.FC<TourProps> = ({ steps, onComplete, onSkip }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  const step = steps[currentStep];

  useLayoutEffect(() => {
    const updateRect = () => {
      const element = document.getElementById(step.targetId);
      if (element) {
        const newRect = element.getBoundingClientRect();
        // Only update if rect has actually changed to avoid loops
        setRect(prev => {
            if (!prev) return newRect;
            if (prev.top !== newRect.top || prev.left !== newRect.left || prev.width !== newRect.width) return newRect;
            return prev;
        });
      }
    };

    const element = document.getElementById(step.targetId);
    if (element) {
       element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Short delay to allow scroll to settle
    const timeout = setTimeout(updateRect, 500);
    
    window.addEventListener('resize', updateRect);
    window.addEventListener('scroll', updateRect, true);
    
    return () => {
      clearTimeout(timeout);
      window.removeEventListener('resize', updateRect);
      window.removeEventListener('scroll', updateRect, true);
    };
  }, [currentStep, step.targetId]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      onComplete();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) setCurrentStep(prev => prev - 1);
  };

  if (!rect) return null;

  const isTop = rect.top > window.innerHeight / 2;

  return (
    <div className="fixed inset-0 z-[9999] pointer-events-none font-sans text-left">
       {/* Top Mask */}
       <div className="absolute bg-black/70 transition-all duration-300 pointer-events-auto" 
            style={{ top: 0, left: 0, right: 0, height: rect.top }} />
       {/* Bottom Mask */}
       <div className="absolute bg-black/70 transition-all duration-300 pointer-events-auto" 
            style={{ top: rect.bottom, left: 0, right: 0, bottom: 0 }} />
       {/* Left Mask */}
       <div className="absolute bg-black/70 transition-all duration-300 pointer-events-auto" 
            style={{ top: rect.top, left: 0, width: rect.left, height: rect.height }} />
       {/* Right Mask */}
       <div className="absolute bg-black/70 transition-all duration-300 pointer-events-auto" 
            style={{ top: rect.top, left: rect.right, right: 0, height: rect.height }} />
            
       {/* Blocker for the hole (prevents interaction with element during tour) */}
       <div className="absolute bg-transparent pointer-events-auto cursor-default"
            style={{ top: rect.top, left: rect.left, width: rect.width, height: rect.height }} />

       {/* Highlight Border */}
       <div className="absolute border-2 border-yellow-400 shadow-[0_0_20px_rgba(250,204,21,0.6)] transition-all duration-300 pointer-events-none"
            style={{ top: rect.top - 4, left: rect.left - 4, width: rect.width + 8, height: rect.height + 8 }} />

       {/* Tooltip */}
       <div className="absolute left-0 right-0 flex justify-center pointer-events-auto"
            style={{ 
                top: isTop ? (rect.top - 24) : (rect.bottom + 24),
                transform: isTop ? 'translateY(-100%)' : 'none'
            }}
       >
          <div className="bg-white border-2 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-6 w-[320px] md:w-[400px] animate-in zoom-in-95 duration-300 relative">
             {/* Connector Triangle */}
             <div className={`absolute left-1/2 -translate-x-1/2 w-4 h-4 bg-white border-l-2 border-t-2 border-black transform rotate-45 ${
                 isTop ? 'bottom-[-9px] border-l-0 border-t-0 border-r-2 border-b-2' : 'top-[-9px]'
             }`}></div>

             <div className="relative z-10">
                <div className="flex justify-between items-center mb-4">
                    <span className="bg-black text-white text-[10px] font-mono font-bold px-2 py-1 uppercase tracking-widest">
                    Tour Guide: {currentStep + 1}/{steps.length}
                    </span>
                    <button onClick={onSkip} className="text-gray-400 hover:text-black transition-colors"><X size={16}/></button>
                </div>
                <h3 className="font-black font-mono text-xl mb-3 uppercase tracking-tight">{step.title}</h3>
                <p className="text-sm font-mono text-gray-600 mb-8 leading-relaxed">{step.content}</p>
                <div className="flex justify-between items-center pt-4 border-t-2 border-gray-100">
                    <button 
                        onClick={handlePrev}
                        disabled={currentStep === 0}
                        className="text-xs font-bold font-mono disabled:opacity-30 hover:underline uppercase"
                    >
                        Back
                    </button>
                    <button 
                        onClick={handleNext}
                        className="bg-yellow-400 text-black px-5 py-2 font-mono font-bold text-xs flex items-center gap-2 border-2 border-black hover:bg-black hover:text-white transition-all shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[1px] hover:translate-y-[1px]"
                    >
                        {currentStep === steps.length - 1 ? 'FINISH' : 'NEXT'} <ChevronRight size={14} />
                    </button>
                </div>
             </div>
          </div>
       </div>
    </div>
  );
};

export default Tour;