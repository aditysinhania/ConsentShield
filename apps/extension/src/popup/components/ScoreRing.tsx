import { riskColor, riskLabel } from "../../shared/risk";

interface Props {
  score: number;
  size?: number;
}

export function ScoreRing({ score, size = 132 }: Props) {
  const stroke = 10;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, score));
  const offset = c - (clamped / 100) * c;
  const color = riskColor(clamped);

  return (
    <div className="score-ring" style={{ width: size, height: size }} role="img" aria-label={`Risk score ${Math.round(clamped)}, ${riskLabel(clamped)}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--border)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="score-ring-label">
        <strong style={{ color }}>{Math.round(clamped)}</strong>
        <span>Risk</span>
      </div>
    </div>
  );
}
