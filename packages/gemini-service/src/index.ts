import { GoogleGenAI, Type, Schema, Content } from '@google/genai';
import { GeminiServiceConfig, AnalysisRequest, AnalysisResult, Question } from './types';

/**
 * GeminiService class for interacting with Google's Gemini AI
 */
export class GeminiService {
    private client: GoogleGenAI;
    private model: string;

    constructor(config: GeminiServiceConfig) {
        this.client = new GoogleGenAI({ apiKey: config.apiKey });
        this.model = config.model || 'gemini-2.5-flash';
    }

    private readonly analysisSchema: Schema = {
        type: Type.OBJECT,
        properties: {
            grade: {
                type: Type.NUMBER,
                description: "A score from 0 to 100 based on correctness, efficiency, and clean code.",
            },
            isCorrect: {
                type: Type.BOOLEAN,
                description: "Whether the code functionally solves the problem correctly.",
            },
            timeComplexityFeedback: {
                type: Type.STRING,
                description: "Feedback on the user's estimated time complexity vs actual.",
            },
            spaceComplexityFeedback: {
                type: Type.STRING,
                description: "Feedback on the user's estimated space complexity vs actual.",
            },
            codeQualityFeedback: {
                type: Type.STRING,
                description: "Comments on code style, variable naming, and best practices.",
            },
            suggestions: {
                type: Type.STRING,
                description: "Specific improvements or a more optimal approach if applicable.",
            }
        },
        required: ["grade", "isCorrect", "timeComplexityFeedback", "spaceComplexityFeedback", "codeQualityFeedback", "suggestions"],
    };

    /**
     * Analyze a coding solution
     */
    async analyzeSolution(request: AnalysisRequest): Promise<AnalysisResult> {
        const prompt = `
      You are a strict technical interviewer at a top tech company. 
      Analyze the following solution for the coding problem "${request.questionTitle}".
      
      Problem Description:
      ${request.questionDescription}
      
      User's Code:
      ${request.userCode}
      
      User's Proposed Time Complexity: ${request.userTimeComplexity}
      User's Proposed Space Complexity: ${request.userSpaceComplexity}
      
      Provide a structured analysis. Be constructive but identify all bugs and inefficiencies.
    `;

        try {
            const response = await this.client.models.generateContent({
                model: this.model,
                contents: prompt,
                config: {
                    responseMimeType: 'application/json',
                    responseSchema: this.analysisSchema,
                    systemInstruction: "You are an expert algorithm instructor. Grade fairly based on correctness and optimality.",
                },
            });

            const text = response.text;
            if (!text) throw new Error("No response from Gemini");

            return JSON.parse(text) as AnalysisResult;
        } catch (error) {
            console.error("Error analyzing solution:", error);
            throw error;
        }
    }

    /**
     * Generate an official solution for a problem
     */
    async generateOfficialSolution(title: string, description: string): Promise<string> {
        const prompt = `
      You are an expert software engineer solving a coding interview problem.
      Problem Title: "${title}"
      Problem Description:
      ${description}

      Please provide the official solution in the following format:
      1. Approach: A concise explanation of the algorithm.
      2. Time Complexity: Big O notation.
      3. Space Complexity: Big O notation.
      4. Code: An optimized Python implementation inside a markdown code block.

      Format Example:
      Approach: Use a Hash Map to store visited elements...
      Time Complexity: O(n)
      Space Complexity: O(n)

      \`\`\`python
      def solve(nums):
          # implementation
      \`\`\`
    `;

        try {
            const response = await this.client.models.generateContent({
                model: this.model,
                contents: prompt,
            });
            return response.text || "Solution generation failed.";
        } catch (error) {
            console.error("Error generating solution:", error);
            return "// Unable to generate solution due to an API error.\\n// Please rely on the analysis feature.";
        }
    }

    /**
     * Get a detailed explanation of a problem
     */
    async getProblemExplanation(question: Question): Promise<string> {
        const prompt = `
      Explain the coding problem "${question.title}" following these exact steps:
      
      Problem Description:
      ${question.description}
      
      Constraints:
      ${question.constraints?.join('\\n') || 'None'}
      
      1. Detailed Explanation: Explain the goal of the problem, inputs, outputs, and constraints clearly.
      2. Logic behind Official Solution: Break down the approach and key steps for the optimal solution.
      3. Algorithmic Hint: Provide a hint towards the algorithmic approach (e.g. "Use a Hash Map") without writing the full code.
    `;

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: prompt,
        });

        return response.text || "Could not generate explanation.";
    }

    /**
     * Identify the coding pattern for a problem
     */
    async identifyCodingPattern(problemDescription: string): Promise<string> {
        const prompt = `
      Analyze the following coding problem description and identify the most likely algorithmic pattern (e.g., Sliding Window, Two Pointers, BFS, DFS, Dynamic Programming, Top K Elements, etc.).
      
      Problem: "${problemDescription}"
      
      Output format (Markdown):
      **Pattern:** [Name of Pattern]
      **Why:** [Brief explanation of the keywords or constraints that give it away]
      **How to Solve:** [1-2 sentences on the standard approach for this pattern]
    `;

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: prompt,
        });

        return response.text || "Could not identify pattern.";
    }

    /**
     * Chat with the coding tutor
     */
    async chatWithTutor(
        history: { role: 'user' | 'model', text: string }[],
        newMessage: string,
        question: Question
    ): Promise<string> {
        const systemInstruction = `
      You are a helpful and encouraging Coding Tutor. 
      The user is working on the problem: "${question.title}".
      
      Problem Description: ${question.description}
      Official Solution Approach: ${question.officialSolution}
      
      Answer the user's questions. 
      - If they ask for a hint, give a small nudge.
      - If they ask about the solution, explain the logic clearly.
      - If they are stuck on specific syntax or logic, guide them.
      - Be concise and friendly.
    `;

        const contents: Content[] = history.map(msg => ({
            role: msg.role,
            parts: [{ text: msg.text }]
        }));

        contents.push({
            role: 'user',
            parts: [{ text: newMessage }]
        });

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: contents,
            config: {
                systemInstruction: systemInstruction,
            }
        });

        return response.text || "I'm having trouble answering that right now.";
    }

    /**
     * Chat with system design tutor
     */
    async chatWithSystemDesignTutor(
        history: { role: 'user' | 'model', text: string }[],
        newMessage: string,
        question: Question
    ): Promise<string> {
        const systemInstruction = `
      You are a Senior Principal Software Architect acting as a System Design Mentor.
      The user is designing: "${question.title}".
      
      Problem Description: ${question.description}
      Official Architecture Summary: ${question.officialSolution}
      
      Your Goal:
      - Engage in a high-level discussion about scalability, availability, and reliability.
      - If the user asks "How do I start?", guide them to requirements gathering (functional vs non-functional).
      - If the user suggests a technology (e.g., "I'll use MySQL"), ask about trade-offs (e.g., "How does that scale for writes vs reads?").
      - Encourage back-of-the-envelope calculations for storage/bandwidth.
      - Be professional, insightful, and challenge the user's assumptions constructively.
      
      Keep responses concise (under 150 words) unless explaining a complex concept.
    `;

        const contents: Content[] = history.map(msg => ({
            role: msg.role,
            parts: [{ text: msg.text }]
        }));

        contents.push({
            role: 'user',
            parts: [{ text: newMessage }]
        });

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: contents,
            config: {
                systemInstruction: systemInstruction,
            }
        });

        return response.text || "I cannot process that architectural query right now.";
    }

    /**
     * Generate a learning module for a topic
     */
    async generateLearningModule(topic: string): Promise<string> {
        const prompt = `
      The user wants to learn about the Computer Science concept: "${topic}".
      
      Create a comprehensive, structured learning module formatted in Markdown.
      Target audience: Software Engineers preparing for technical interviews.
      
      Structure the response exactly as follows:
      
      # ${topic}
      
      ## 1. Concept Overview
      (A clear, high-level definition of what it is and why it matters)
      
      ## 2. How It Works
      (A detailed explanation of the mechanics, logic, or data structure visualization)
      
      ## 3. Implementation / Pseudocode
      (Provide a code block, preferably in Python, demonstrating the concept)
      
      ## 4. Complexity Analysis
      (Time and Space complexity with explanation)
      
      ## 5. Example Problem
      (A LeetCode-style problem statement that requires this concept)
      
      ### Problem Description
      ...
      
      ### Walkthrough
      (Step-by-step application of the concept to solve this problem)
    `;

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: prompt,
        });

        return response.text || "Unable to generate learning module.";
    }

    /**
     * Chat with learning tutor
     */
    async chatWithLearningTutor(
        history: { role: 'user' | 'model', text: string }[],
        newMessage: string,
        topic: string
    ): Promise<string> {
        const systemInstruction = `
      You are an expert Computer Science Professor.
      The user is currently studying the topic: "${topic}".
      
      Your goal is to answer their follow-up questions, clarify doubts, or provide more examples related to ${topic}.
      Be educational, rigorous, yet encouraging.
      If they ask for code, provide it in Python.
    `;

        const contents: Content[] = history.map(msg => ({
            role: msg.role,
            parts: [{ text: msg.text }]
        }));

        contents.push({
            role: 'user',
            parts: [{ text: newMessage }]
        });

        const response = await this.client.models.generateContent({
            model: this.model,
            contents: contents,
            config: {
                systemInstruction: systemInstruction,
            }
        });

        return response.text || "I cannot answer that right now.";
    }
}

export type { Content, AnalysisRequest, AnalysisResult, Question, GeminiServiceConfig };
