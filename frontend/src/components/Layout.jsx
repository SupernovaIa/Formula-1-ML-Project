import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/race-report", icon: "🏁", label: "Race Weekend" },
  { to: "/season-report", icon: "🏆", label: "Championship" },
  { to: "/circuit-clustering", icon: "🧭", label: "Circuits" },
  { to: "/winner-prediction", icon: "🔮", label: "Race Predictor" },
  { to: "/chatbot", icon: "💬", label: "Chatbot" },
];

export default function Layout() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-inner">
          <NavLink to="/" className="app-title" end>
            🏎️ Formula 1 ML
          </NavLink>
          <nav className="app-nav">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
              >
                <span aria-hidden="true">{item.icon}</span> {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  );
}
