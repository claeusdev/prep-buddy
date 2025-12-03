# @prepbuddy/storage

A TypeScript library for managing user progress with localStorage.

## Installation

```bash
npm install @prepbuddy/storage
```

## Usage

```typescript
import { StorageService } from '@prepbuddy/storage';

// Initialize the service
const storage = new StorageService({
  storageKey: 'my_app_progress' // optional, defaults to 'prepbuddy_progress_v1'
});

// Save progress for a question
await storage.saveProgress('question-123', {
  isSolved: true,
  grade: 95,
  userCode: 'def solution(): ...',
  analysisResult: { /* ... */ }
});

// Get progress for a specific question
const progress = await storage.getProgress('question-123');

// Get all progress
const allProgress = await storage.getAllProgress();

// Clear all progress
await storage.clearAllProgress();
```

## Features

- ✅ Local storage for progress tracking
- ✅ Question progress tracking
- ✅ Code and chat history storage
- ✅ Diagram/whiteboard data persistence
- ✅ TypeScript support

## API

### `getProgress(questionId: string): Promise<QuestionProgress>`
Retrieves progress for a specific question from localStorage.

### `getAllProgress(): Promise<Record<string, QuestionProgress>>`
Retrieves all stored progress across all questions.

### `saveProgress(questionId: string, data: Partial<QuestionProgress>): Promise<void>`
Saves or updates progress for a question. Automatically merges with existing data.

### `clearAllProgress(): Promise<void>`
Clears all progress from localStorage.

## Types

### `QuestionProgress`
```typescript
interface QuestionProgress {
  isSolved: boolean;
  grade: number;
  explanation?: string;
  chatHistory?: ChatMessage[];
  userCode?: string;
  diagramData?: string;
  timestamp?: number;
  analysisResult?: AnalysisResult;
}
```

## License

MIT
