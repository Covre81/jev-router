export type BreakerState = 'closed' | 'open' | 'half-open';

export class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private customCooldownMs?: number | undefined;
  public inFlight = 0;

  constructor(
    private readonly maxFailures: number,
    private readonly cooldownMs: number,
    private readonly maxConcurrency: number,
    private readonly clock: () => number = Date.now,
  ) {}

  get state(): BreakerState {
    if (this.failures >= this.maxFailures) {
      const cooldown = this.customCooldownMs ?? this.cooldownMs;
      if (this.clock() - this.lastFailureTime > cooldown) {
        return 'half-open';
      }
      return 'open';
    }
    return 'closed';
  }

  forceOpen(cooldownMs: number): void {
    this.failures = this.maxFailures;
    this.lastFailureTime = this.clock();
    this.customCooldownMs = cooldownMs;
  }

  acquire(): boolean {
    const s = this.state;
    if (s === 'open') return false;
    if (s === 'half-open' && this.inFlight > 0) return false; // Only one trial in half-open
    if (this.inFlight >= this.maxConcurrency) return false;
    
    this.inFlight++;
    return true;
  }

  release(outcome: 'success' | 'failure' | 'neutral'): void {
    this.inFlight--;
    if (outcome === 'neutral') return;

    if (outcome === 'success') {
      this.failures = 0;
      this.lastFailureTime = 0;
      this.customCooldownMs = undefined;
    } else {
      if (this.failures >= this.maxFailures) {
        // already open/half-open, push/keep open
        this.failures = this.maxFailures + 1;
      } else {
        this.failures++;
      }
      this.lastFailureTime = this.clock();
    }
  }
}
