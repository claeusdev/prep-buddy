# @prepbuddy/gemini-service

A TypeScript library for interacting with Google's Gemini AI API with a focus on coding education and interview preparation.

## Installation

```bash
npm install @prepbuddy/gemini-service
```

## Usage

```typescript
import { GeminiService } from '@prepbuddy/gemini-service';

// Initialize the service
const gemini = new GeminiService({
  apiKey: 'your-gemini-api-key',
  model: 'gemini-2.5-flash' // optional, defaults to gemini-2.5-flash
});

// Analyze a coding solution
const analysis = await gemini.analyzeSolution({
  questionTitle: 'Two Sum',
  questionDescription: 'Find two numbers that add up to target...',
  userCode: 'def twoSum(nums, target): ...',
  userTimeComplexity: 'O(n)',
  userSpaceComplexity: 'O(n)'
});

console.log(analysis.grade, analysis.suggestions);

// Generate official solution
const solution = await gemini.generateOfficialSolution(
  'Two Sum',
  'Given an array of integers...'
);

// Chat with coding tutor
const response = await gemini.chatWithTutor(
  chatHistory,
  'Can you give me a hint?',
  question
);
```

## Features

- ✅ Solution analysis with grading
- ✅ Official solution generation
- ✅ Problem explanation
- ✅ Coding pattern identification
- ✅ Interactive coding tutor chat
- ✅ System design mentor chat
- ✅ Learning module generation
- ✅ Full TypeScript support

## API

### `analyzeSolution(request: AnalysisRequest): Promise<AnalysisResult>`
Analyzes a user's code solution and provides feedback.

### `generateOfficialSolution(title: string, description: string): Promise<string>`
Generates an optimal solution for a given problem.

### `getProblemExplanation(question: Question): Promise<string>`
Provides a detailed explanation of a coding problem.

### `identifyCodingPattern(problemDescription: string): Promise<string>`
Identifies the algorithmic pattern needed to solve a problem.

### `chatWithTutor(history, newMessage, question): Promise<string>`
Interactive chat with an AI coding tutor.

### `chatWithSystemDesignTutor(history, newMessage, question): Promise<string>`
Interactive chat with an AI system design mentor.

### `generateLearningModule(topic: string): Promise<string>`
Generates a comprehensive learning module for a CS topic.

### `chatWithLearningTutor(history, newMessage, topic): Promise<string>`
Interactive chat with an AI learning tutor.

## License

MIT
