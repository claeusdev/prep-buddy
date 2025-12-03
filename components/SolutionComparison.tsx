
import React from 'react';
import Editor from './Editor';
import { Code2, FileCheck } from 'lucide-react';

interface SolutionComparisonProps {
  userCode: string;
  officialSolutionMarkdown: string;
}

const SolutionComparison: React.FC<SolutionComparisonProps> = ({ userCode, officialSolutionMarkdown }) => {
  
  // Extract python code block from markdown
  const officialCode = React.useMemo(() => {
    const match = officialSolutionMarkdown.match(/```python([\s\S]*?)```/);
    return match ? match[1].trim() : officialSolutionMarkdown;
  }, [officialSolutionMarkdown]);

  return (
    <div className="h-[70vh] flex flex-col">
      <div className="flex-1 flex flex-col md:flex-row border border-gray-200 rounded-xl overflow-hidden">
        
        {/* Left: User Code */}
        <div className="flex-1 flex flex-col border-b md:border-b-0 md:border-r border-gray-200">
          <div className="bg-gray-50 p-3 border-b border-gray-200 flex items-center gap-2">
            <Code2 size={16} className="text-blue-600" />
            <span className="text-sm font-bold text-gray-700">Your Solution</span>
          </div>
          <div className="flex-1 relative overflow-hidden bg-white">
             <div className="absolute inset-0 overflow-y-auto">
               <Editor code={userCode} setCode={() => {}} /> {/* Read Only essentially since we ignore setCode */}
             </div>
          </div>
        </div>

        {/* Right: Official Code */}
        <div className="flex-1 flex flex-col bg-slate-50">
           <div className="bg-gray-50 p-3 border-b border-gray-200 flex items-center gap-2">
             <FileCheck size={16} className="text-green-600" />
             <span className="text-sm font-bold text-gray-700">Official Solution</span>
           </div>
           <div className="flex-1 relative overflow-hidden bg-white">
             <div className="absolute inset-0 overflow-y-auto">
               <Editor code={officialCode} setCode={() => {}} />
             </div>
           </div>
        </div>

      </div>
      <div className="mt-4 text-center text-xs text-gray-500">
        Compare your logic and syntax with the reference implementation.
      </div>
    </div>
  );
};

export default SolutionComparison;
