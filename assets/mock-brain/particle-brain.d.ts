export interface BrainState {
  mode: 'hasbro' | 'spar';
  generating: boolean;
  progress: number | null;
  deviations: [number, number, number];
  activity: [number, number, number];
  paused: boolean;
  yaw: number;
  pitch: number;
}
export type BrainPatch = Partial<Pick<BrainState, 'mode' | 'generating' | 'progress' | 'deviations' | 'activity' | 'paused'>> & { deviation?: number };
export interface BrainController {
  element: HTMLDivElement;
  setState(patch?: BrainPatch): void;
  getState(): BrainState;
  setView(view?: { yaw?: number; pitch?: number }): void;
  refreshTheme(): void;
  destroy(): void;
}
export function createParticleBrain(container: Element, options?: BrainPatch & { controls?: boolean }): BrainController;
