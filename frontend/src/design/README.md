# design/

Copy of the **Atlas / Blueprint** design system from the personal Life OS
(`docs/personal-life-os/design/`, adopted 2026-10-08). Source of truth lives
there; this folder is a snapshot so the repo builds on its own.

| File | Origin |
|---|---|
| `tokens.css`, `components.css` | verbatim copy - don't edit here, change Atlas and re-copy |
| `fonts.css` + `public/fonts/*.woff2` | same families as Atlas, served as files instead of data URIs |
| `icons/*.svg` | Lucide subset, only the icons the app uses (see `components/Icon.jsx`) |
| `f1.css` | **project-only** additions (extra categorical colours for 12 circuit clusters) |

Density used here is Atlas's **Instrument** one (dashboard): serif only in the
masthead and KPI figures, no scroll reveals, no chapter rail. Rules: Atlas `DESIGN.md`.
