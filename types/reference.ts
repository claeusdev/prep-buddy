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

