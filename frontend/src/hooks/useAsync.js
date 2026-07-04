import { useEffect, useState } from "react";

// Runs `fetcher` whenever `deps` change. Pass `enabled=false` to skip
// (e.g. before the user has picked required inputs).
export function useAsync(fetcher, deps, enabled = true) {
  const [state, setState] = useState({ data: null, loading: enabled, error: null });

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null });
      return;
    }

    let cancelled = false;
    setState((prev) => ({ ...prev, loading: true, error: null }));

    fetcher()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error) => {
        if (!cancelled) setState({ data: null, loading: false, error });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
