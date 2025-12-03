import { StorageServiceConfig, QuestionProgress } from "./types";


/**
 * StorageService - Handles progress storage with localStorage
 */
export class StorageService {
    private storageKey: string;

    constructor(config: StorageServiceConfig = {}) {
        this.storageKey = config.storageKey || 'prepbuddy_progress_v1';
    }

    /**
     * Retrieve progress for a specific question
     */
    async getProgress(questionId: string): Promise<QuestionProgress> {
        try {
            const raw = localStorage.getItem(this.storageKey);
            if (!raw) return { isSolved: false, grade: 0 };

            const allData = JSON.parse(raw);
            return allData[questionId] || { isSolved: false, grade: 0 };
        } catch (e) {
            console.error("Failed to load progress", e);
            return { isSolved: false, grade: 0 };
        }
    }

    /**
     * Retrieve all progress (for SavedSolutions or dashboard)
     */
    async getAllProgress(): Promise<Record<string, QuestionProgress>> {
        try {
            const raw = localStorage.getItem(this.storageKey);
            if (!raw) return {};
            return JSON.parse(raw);
        } catch (e) {
            console.error("Failed to load all progress", e);
            return {};
        }
    }

    /**
     * Save specific fields for a question
     */
    async saveProgress(questionId: string, data: Partial<QuestionProgress>): Promise<void> {
        try {
            // Prepare update object
            const updates = {
                ...data,
                timestamp: data.isSolved ? Date.now() : (data.timestamp || Date.now())
            };

            const raw = localStorage.getItem(this.storageKey);
            const allData = raw ? JSON.parse(raw) : {};
            const current = allData[questionId] || { isSolved: false, grade: 0 };

            // Merge with existing local state
            allData[questionId] = { ...current, ...updates };

            localStorage.setItem(this.storageKey, JSON.stringify(allData));
        } catch (e) {
            console.error("Failed to save progress", e);
        }
    }

    /**
     * Clear all progress (useful for testing or reset functionality)
     */
    async clearAllProgress(): Promise<void> {
        try {
            localStorage.removeItem(this.storageKey);
        } catch (e) {
            console.error("Failed to clear progress", e);
        }
    }
}
