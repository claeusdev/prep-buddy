
import React from 'react';
import CodeEditor from 'react-simple-code-editor';
import Prism from 'prismjs';

interface EditorProps {
  code: string;
  setCode: (code: string) => void;
}

const Editor: React.FC<EditorProps> = ({ code, setCode }) => {
  const highlight = (code: string) => {
    return Prism.highlight(
      code, 
      Prism.languages.javascript || Prism.languages.clike, 
      'javascript'
    );
  };

  return (
    <div className="w-full h-full flex flex-col bg-white overflow-hidden relative group">
      {/* Line numbers background simulation */}
      <div className="absolute left-0 top-0 bottom-0 w-12 bg-gray-50 border-r border-gray-200 z-0"></div>
      
      <div className="flex-1 relative overflow-auto z-10 pl-2">
         <CodeEditor
            value={code}
            onValueChange={code => setCode(code)}
            highlight={highlight}
            padding={24}
            placeholder="// Initialize solution..."
            className="prism-editor font-mono text-sm min-h-full"
            style={{
              fontFamily: '"Fira Code", monospace',
              fontSize: 14,
              backgroundColor: 'transparent',
              minHeight: '100%',
            }}
            textareaClassName="focus:outline-none"
          />
      </div>
    </div>
  );
};

export default Editor;
