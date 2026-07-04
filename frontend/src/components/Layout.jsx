import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/race-report", label: "Race Report" },
  { to: "/season-report", label: "Season Report" },
  { to: "/circuit-clustering", label: "Circuit Clustering" },
  { to: "/winner-prediction", label: "Winner Prediction" },
];

export default function Layout() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-title">🏎️ Formula 1 ML Project</span>
        <nav className="app-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  );
}
