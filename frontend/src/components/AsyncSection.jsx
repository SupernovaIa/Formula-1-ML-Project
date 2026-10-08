import Icon from "./Icon";

export default function AsyncSection({ loading, error, children }) {
  if (loading) {
    return (
      <p className="status status--loading" role="status">
        <span className="loading-dot" />
        Loading…
      </p>
    );
  }
  if (error) {
    return (
      <p className="status status--error" role="alert">
        <Icon name="circle-alert" />
        {error.message}
      </p>
    );
  }
  return children;
}
