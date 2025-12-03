export interface AnalysisRequest {
    questionTitle: string;
    questionDescription: string;
    userCode: string;
    userTimeComplexity: string;
    userSpaceComplexity: string;
}

export interface AnalysisResult {
    grade: number;
    isCorrect: boolean;
    timeComplexityFeedback: string;
    spaceComplexityFeedback: string;
    codeQualityFeedback: string;
    suggestions: string;
}

export interface Question {
    title: string;
    description: string;
    constraints?: string[];
    officialSolution?: string;
}

export interface GeminiServiceConfig {
    apiKey: string;
    model?: string;
}
