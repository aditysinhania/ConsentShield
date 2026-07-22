export const SCAN_STEPS = [
  "Scanning page…",
  "Extracting DOM…",
  "Detecting consent banner…",
  "Running Rule Engine…",
  "Running NLP…",
  "Running Vision…",
  "Combining results…",
  "Generating report…",
] as const;

interface Props {
  activeIndex: number;
  message?: string;
}

export function ProgressSteps({ activeIndex, message }: Props) {
  const idx = Math.max(0, Math.min(SCAN_STEPS.length - 1, activeIndex));
  return (
    <div className="progress-panel card fade-in" role="status" aria-live="polite">
      <div className="progress-head">
        <span className="spinner" aria-hidden="true" />
        <strong>{message || SCAN_STEPS[idx]}</strong>
      </div>
      <ol className="progress-list">
        {SCAN_STEPS.map((step, i) => (
          <li key={step} className={i < idx ? "done" : i === idx ? "active" : ""}>
            {step.replace("…", "")}
          </li>
        ))}
      </ol>
      <div className="progress-bar" aria-hidden="true">
        <div style={{ width: `${((idx + 1) / SCAN_STEPS.length) * 100}%` }} />
      </div>
    </div>
  );
}
