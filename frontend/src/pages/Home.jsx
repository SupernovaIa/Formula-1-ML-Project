import { Link } from "react-router-dom";

const SECTIONS = [
  {
    to: "/race-report",
    icon: "🏁",
    title: "Race Weekend",
    description: "Relive any Grand Prix: qualifying, results, pace, tyre strategy, lap-by-lap telemetry.",
  },
  {
    to: "/season-report",
    icon: "🏆",
    title: "Championship",
    description: "Watch a title fight unfold — drivers' and constructors' standings, race by race.",
  },
  {
    to: "/circuit-clustering",
    icon: "🧭",
    title: "Circuits",
    description: "See which tracks drive alike — fast and flowing, tight and technical, or somewhere in between.",
  },
  {
    to: "/winner-prediction",
    icon: "🔮",
    title: "Race Predictor",
    description: "Pick a race and driver, see what our model would have called it — then try a different grid slot.",
  },
  {
    to: "/chatbot",
    icon: "💬",
    title: "Chatbot",
    description: "Ask questions about a Grand Prix and get answers grounded in the real race data.",
  },
];

const STATS = [
  { value: "97.4%", label: "Winner prediction accuracy" },
  { value: "7", label: "Circuit types" },
  { value: "2010–24", label: "Seasons covered" },
  { value: "30", label: "Circuits studied" },
];

export default function Home() {
  return (
    <div className="page home">
      <div className="home-hero">
        <h1>Formula 1, explored</h1>
        <p className="home-tagline">
          Relive race weekends, track championship battles, explore how
          circuits differ, and see what our model makes of it all — built on
          real F1 timing and telemetry data.
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
