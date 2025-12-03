import type { ChatMessage } from './chat';
import type { AnalysisResult } from './question';

export interface QuestionProgress {
  isSolved: boolean;
  grade: number;
  explanation?: string;
  chatHistory?: ChatMessage[];
  userCode?: string;
  diagramData?: string; // JSON string of Whiteboard elements
  timestamp?: number;
  analysisResult?: AnalysisResult;
}

export interface StorageServiceConfig {
  storageKey?: string;
}

