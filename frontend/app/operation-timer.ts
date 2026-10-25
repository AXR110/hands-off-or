export type OperationTimer = {
  startedAt: number | null;
  stoppedAt: number | null;
};

export function idleOperationTimer(): OperationTimer {
  return { startedAt: null, stoppedAt: null };
}

/** The first wake starts the clock. Later wakes and a stopped clock leave it unchanged. */
export function startOperationTimer(timer: OperationTimer, now: number): OperationTimer {
  if (timer.startedAt != null || timer.stoppedAt != null) return timer;
  return { startedAt: now, stoppedAt: null };
}

export function stopOperationTimer(timer: OperationTimer, now: number): OperationTimer {
  if (timer.startedAt == null || timer.stoppedAt != null) return timer;
  return { startedAt: timer.startedAt, stoppedAt: now };
}

export function operationElapsedMs(timer: OperationTimer, now: number): number {
  if (timer.startedAt == null) return 0;
  const end = timer.stoppedAt ?? now;
  return Math.max(0, end - timer.startedAt);
}

export function formatOperationDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}
