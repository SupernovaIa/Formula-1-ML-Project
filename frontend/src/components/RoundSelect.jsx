import { useEffect } from "react";
import { useAsync } from "../hooks/useAsync";
import { getSeasonRounds } from "../api/client";
import { humanizeSlug } from "../utils/format";

// A season's rounds, labeled by circuit instead of a bare number - shared by
// any page that lets a user pick "which race" (Race Weekend, Race Predictor).
export default function RoundSelect({ year, value, onChange }) {
  const { data: rounds } = useAsync(() => getSeasonRounds(year), [year]);

  useEffect(() => {
    if (rounds?.length && !rounds.some((r) => r.round === value)) {
      onChange(rounds[0].round);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rounds]);

  return (
    <label>
      Race
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {rounds?.map((r) => (
          <option key={r.round} value={r.round}>
            Round {r.round} — {humanizeSlug(r.circuit_id)}
          </option>
        ))}
      </select>
    </label>
  );
}
