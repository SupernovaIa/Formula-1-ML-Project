import { Link } from "react-router-dom";

const SECTIONS = [
  {
    to: "/race-report",
    icon: "🏁",
    title: "Race Report",
    description: "Qualifying and race breakdowns: results, telemetry, tyre strategy, pace.",
  },
  {
    to: "/season-report",
    icon: "🏆",
    title: "Season Report",
    description: "Drivers' and constructors' championship standings by season.",
  },
  {
    to: "/circuit-clustering",
    icon: "🧭",
    title: "Circuit Clustering",
    description: "K-Means clustering of circuits by technical profile — speed, corners, straights.",
  },
  {
    to: "/winner-prediction",
    icon: "🔮",
    title: "Winner Prediction",
    description: "XGBoost model estimating win probability from grid position and form.",
  },
];

const STATS = [
  { value: "97.4%", label: "Classifier accuracy" },
  { value: "7", label: "Circuit clusters" },
  { value: "2018–24", label: "Seasons covered" },
  { value: "0.967", label: "AUC-ROC" },
];

export default function Home() {
  return (
    <div className="page home">
      <div className="home-hero">
        <h1>Formula 1 ML Project</h1>
        <p className="home-tagline">
          EDA, circuit clustering and race-winner prediction on top of FastF1 data —
          served through a FastAPI backend.
        </p>
      </div>

      <div className="stat-strip">
        {STATS.map((s) => (
          <div className="stat" key={s.label}>
            <span className="stat-value">{s.value}</span>
            <span className="stat-label">{s.label}</span>
          </div>
        ))}
      </div>

      <div className="section-grid">
        {SECTIONS.map((s) => (
          <Link to={s.to} className="section-card" key={s.to}>
            <span className="section-icon">{s.icon}</span>
            <span className="section-title">{s.title}</span>
            <span className="section-description">{s.description}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
