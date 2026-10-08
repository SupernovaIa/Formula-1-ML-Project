import { NavLink, Outlet } from "react-router-dom";
import Icon from "./Icon";
import { toggleTheme } from "../hooks/useThemeMode";

const SECTIONS = [
  { to: "/race-report", icon: "flag", label: "Race Weekend" },
  { to: "/season-report", icon: "award", label: "Championship" },
  { to: "/circuit-clustering", icon: "compass", label: "Circuits" },
  { to: "/winner-prediction", icon: "target", label: "Race Predictor" },
  { to: "/chatbot", icon: "bot", label: "Chatbot" },
];

export default function Layout() {
  return (
    <>
      <header className="a-topbar">
        <NavLink to="/" end className="a-brand">
          <b>Formula 1</b> ML
        </NavLink>
        <nav className="app-nav" aria-label="Sections">
          {SECTIONS.map((s) => (
            <NavLink key={s.to} to={s.to} className="app-nav-link">
              <Icon name={s.icon} />
              <span>{s.label}</span>
            </NavLink>
          ))}
        </nav>
        <button type="button" className="a-icon-btn" aria-label="Toggle light / dark theme" onClick={toggleTheme}>
          <Icon name="moon" className="a-theme-moon" />
          <Icon name="sun" className="a-theme-sun" />
        </button>
      </header>
      <main className="app-content a-wrap">
        <Outlet />
      </main>
    </>
  );
}
