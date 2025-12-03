
export interface Question {
  id: string;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  category: string;
  description: string;
  examples?: { input: string; output: string; explanation?: string }[];
  constraints?: string[];
  officialSolution: string; // The "Back" of the card
  tags?: string[];
  companies?: string[];
}

export interface AnalysisResult {
  grade: number;
  isCorrect: boolean;
  timeComplexityFeedback: string;
  spaceComplexityFeedback: string;
  codeQualityFeedback: string;
  suggestions: string;
}

export interface AnalysisRequest {
  questionTitle: string;
  questionDescription: string;
  userCode: string;
  userTimeComplexity: string;
  userSpaceComplexity: string;
}

export interface ReferenceItem {
  id: string;
  title: string;
  category: 'Data Structure' | 'Algorithm' | 'Concept' | 'System Design';
  summary: string;
  complexity: {
    time: string;
    space: string;
  };
  description: string;
  implementation: string; // Python code
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}
