// Types matching the shared types folder structure
export interface ChatMessage {
    role: 'user' | 'model';
    text: string;
}

export interface AnalysisResult {
    grade: number;
    isCorrect: boolean;
    timeComplexityFeedback: string;
    spaceComplexityFeedback: string;
    codeQualityFeedback: string;
    suggestions: string;
}

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