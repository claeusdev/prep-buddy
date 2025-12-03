import { GeminiService } from '@prepbuddy/gemini-service';
import { StorageService } from '@prepbuddy/storage';

// Initialize Gemini Service
const geminiService = new GeminiService({
    apiKey: import.meta.env.VITE_GEMINI_API_KEY,
    model: 'gemini-2.5-flash'
});

// Initialize Storage Service
export const storageService = new StorageService();

// Export Gemini methods with backward-compatible function signatures
export const analyzeSolution = geminiService.analyzeSolution.bind(geminiService);
export const generateOfficialSolution = geminiService.generateOfficialSolution.bind(geminiService);
export const getProblemExplanation = geminiService.getProblemExplanation.bind(geminiService);
export const identifyCodingPattern = geminiService.identifyCodingPattern.bind(geminiService);
export const chatWithTutor = geminiService.chatWithTutor.bind(geminiService);
export const chatWithSystemDesignTutor = geminiService.chatWithSystemDesignTutor.bind(geminiService);
export const generateLearningModule = geminiService.generateLearningModule.bind(geminiService);
export const chatWithLearningTutor = geminiService.chatWithLearningTutor.bind(geminiService);

// Export Storage methods with backward-compatible function signatures
export const getStoredProgress = storageService.getProgress.bind(storageService);
export const saveStoredProgress = storageService.saveProgress.bind(storageService);
export const getAllProgress = storageService.getAllProgress.bind(storageService);

// Re-export types for convenience
export type {
    AnalysisRequest,
    AnalysisResult,
    Question,
    GeminiServiceConfig
} from '@prepbuddy/gemini-service';
