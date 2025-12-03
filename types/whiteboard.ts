export interface DiagramElement {
  id: string;
  type: 'rect' | 'database' | 'arrow' | 'text' | 'pencil';
  x: number;
  y: number;
  width?: number;
  height?: number;
  points?: { x: number; y: number }[];
  text?: string;
  color?: string;
}

