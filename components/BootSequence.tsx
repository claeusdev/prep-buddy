
import React, { useState, useEffect } from 'react';

export const BootSequence = ({ onComplete }: { onComplete: () => void }) => {
  const [lines, setLines] = useState<string[]>([]);
  const sequence = [
    "BIOS DATE 01/01/99 14:22:55 VER 1.02",
    "CPU: QUANTUM V20, SPEED: ∞ MHz",
    "640K RAM SYSTEM... 640K OK",
    " ",
    "> LOADING_KERNEL...",
    "> INITIALIZING_VIDEO_ADAPTER...",
    "> MOUNTING_VIRTUAL_DRIVES...",
    "> LOADING_AI_MODULES (GEMINI-2.5-FLASH)... OK",
    "> ESTABLISHING_SECURE_UPLINK...",
    " ",
    "SYSTEM_READY."
  ];

  useEffect(() => {
    let delay = 0;
    sequence.forEach((line, index) => {
      // Randomize delay for typing effect
      delay += Math.random() * 300 + 100; 
      
      setTimeout(() => {
        setLines(prev => [...prev, line]);
        
        // If it's the last line, trigger complete after a short pause
        if (index === sequence.length - 1) {
            setTimeout(onComplete, 1000);
        }
      }, delay);
    });
  }, []);

  return (
    <div className="fixed inset-0 bg-black text-green-500 font-mono p-6 z-[100] text-sm md:text-base overflow-hidden flex flex-col justify-end pb-20 md:justify-start md:pb-0">
        <div className="max-w-2xl">
            {lines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap mb-1">{line}</div>
            ))}
            <div className="animate-pulse mt-2 bg-green-500 w-3 h-5 inline-block"></div>
        </div>
    </div>
  );
};
