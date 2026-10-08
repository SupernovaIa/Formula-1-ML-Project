// Instrument-density masthead: a label, a serif title with one italic word in
// the accent colour (`em`), and an optional lede. One per page.
export default function PageHead({ label, title, em, lede }) {
  return (
    <header className="page-head">
      {label && <span className="a-label">{label}</span>}
      <h1 className="a-display page-title">
        {title} {em && <em>{em}</em>}
      </h1>
      {lede && <p className="a-lede">{lede}</p>}
    </header>
  );
}
