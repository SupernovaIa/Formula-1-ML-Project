export default function AsyncSection({ loading, error, children }) {
  if (loading) return <p className="status-text">Loading...</p>;
  if (error) return <p className="status-text error">{error.message}</p>;
  return children;
}
