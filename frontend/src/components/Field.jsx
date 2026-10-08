// Label above a control. The label is the Atlas `a-label` signature.
export default function Field({ label, children }) {
  return (
    <label className="field">
      <span className="a-label">{label}</span>
      {children}
    </label>
  );
}
