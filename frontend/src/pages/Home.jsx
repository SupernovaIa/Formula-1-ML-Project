import { Link } from "react-router-dom";
import Icon from "../components/Icon";
import PageHead from "../components/PageHead";

const SECTIONS = [
  {
    to: "/race-report",
    icon: "flag",
    title: "Race Weekend",
    description: "Relive any Grand Prix: qualifying, results, pace, tyre strategy, lap-by-lap telemetry.",
  },
  {
    to: "/season-report",
    icon: "award",
    title: "Championship",
    description: "Watch a title fight unfold — drivers' and constructors' standings, race by race.",
  },
  {
    to: "/circuit-clustering",
    icon: "compass",
    title: "Circuits",
    description: "See which tracks drive alike — fast and flowing, tight and technical, or somewhere in between.",
  },
  {
    to: "/winner-prediction",
    icon: "target",
    title: "Race Predictor",
    description: "Pick a race and driver, see what our model would have called it — then try a different grid slot.",
  },
  {
    to: "/chatbot",
    icon: "bot",
    title: "Chatbot",
    description: "Ask questions about a Grand Prix and get answers grounded in the real race data.",
  },
];

export default function Home() {
  return (
    <div className="page">
      <div className="home-hero">
        <PageHead
          title="Formula 1,"
          em="explored"
          lede="Relive race weekends, track championship battles, explore how circuits differ, and see what our model makes of it all — built on real F1 timing and telemetry data."
        />
        <p className="a-facts">
          <span><b>2010–24</b> seasons</span>
          <span><b>30</b> circuits studied</span>
          <span><b>7</b> circuit types</span>
        </p>
      </div>

      {/* The one lit element on the page: the model's headline number, and the way in. */}
      <div className="a-seam home-seam">
        <div>
          <span className="a-kicker">Winner prediction</span>
          <div className="a-stat-value">97.4%</div>
          <p>accuracy of the race-winner model</p>
        </div>
        <Link to="/winner-prediction" className="seam-link">
          Try it on a real race <Icon name="arrow-right" />
        </Link>
      </div>

      <nav className="a-panel section-list" aria-label="Sections">
        {SECTIONS.map((s) => (
          <Link to={s.to} className="section-row" key={s.to}>
            <span className="a-glyph"><Icon name={s.icon} /></span>
            <span className="section-row-body">
              <span className="section-row-title">{s.title}</span>
              <span className="section-row-desc">{s.description}</span>
            </span>
            <Icon name="arrow-up-right" className="section-row-arrow" />
          </Link>
        ))}
      </nav>
    </div>
  );
}
