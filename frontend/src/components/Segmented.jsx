// 2-3 mutually exclusive views (e.g. Drivers / Constructors).
export default function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button
          key={o}
          type="button"
          role="tab"
          aria-selected={value === o}
          className="segmented-btn"
          onClick={() => onChange(o)}
        >
          {o}
        </button>
      ))}
    </div>
  );
}
