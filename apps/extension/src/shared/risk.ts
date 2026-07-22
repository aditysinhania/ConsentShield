/** Risk score color coding — professional palette. */

export type RiskTier = "low" | "moderate" | "elevated" | "high";

export function riskTier(score: number): RiskTier {
  if (score < 25) return "low";
  if (score < 50) return "moderate";
  if (score < 75) return "elevated";
  return "high";
}

export function riskColor(score: number): string {
  switch (riskTier(score)) {
    case "low":
      return "var(--risk-low)";
    case "moderate":
      return "var(--risk-moderate)";
    case "elevated":
      return "var(--risk-elevated)";
    case "high":
      return "var(--risk-high)";
  }
}

export function riskLabel(score: number): string {
  switch (riskTier(score)) {
    case "low":
      return "Low risk";
    case "moderate":
      return "Moderate risk";
    case "elevated":
      return "Elevated risk";
    case "high":
      return "High risk";
  }
}
