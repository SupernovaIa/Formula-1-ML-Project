import { useSyncExternalStore } from "react";

const STORAGE_KEY = "f1-theme";
const query = () => window.matchMedia("(prefers-color-scheme: dark)");

// Explicit data-theme wins; without it the system decides (same rule as Atlas).
function current() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit === "light" || explicit === "dark") return explicit;
  return query().matches ? "dark" : "light";
}

function subscribe(notify) {
  const observer = new MutationObserver(notify);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = query();
  mq.addEventListener("change", notify);
  return () => {
    observer.disconnect();
    mq.removeEventListener("change", notify);
  };
}

export function useThemeMode() {
  return useSyncExternalStore(subscribe, current, () => "light");
}

export function toggleTheme() {
  const next = current() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // private mode / blocked storage: the theme just won't persist
  }
}
