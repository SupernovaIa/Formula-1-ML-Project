import { useEffect, useState } from "react";
import AsyncSection from "../components/AsyncSection";
import RoundSelect from "../components/RoundSelect";
import { useAsync } from "../hooks/useAsync";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { getDriverForm, getRoundEntrants, predictWinner } from "../api/client";
import { humanizeSlug } from "../utils/format";

const YEARS = Array.from({ length: 2024 - 2010 + 1 }, (_, i) => 2010 + i).reverse();

export default function WinnerPrediction() {
  const [year, setYear] = useState(2023);
  const [roundNumber, setRoundNumber] = useState(1);
  const [driverId, setDriverId] = useState("");
  const [gridPosition, setGridPosition] = useState(null);

  const { data: entrants } = useAsync(() => getRoundEntrants(year, roundNumber), [year, roundNumber]);

  useEffect(() => {
    if (entrants?.drivers?.length && !entrants.drivers.includes(driverId)) {
      setDriverId(entrants.drivers[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entrants]);

  const form = useAsync(
    () => getDriverForm(year, roundNumber, driverId),
    [year, roundNumber, driverId],
    Boolean(driverId)
  );

  // Reset the "what if" grid slot to the driver's real one whenever the
  // race/driver picked changes - but not while the user is dragging it.
  useEffect(() => {
    if (form.data) setGridPosition(form.data.grid_position);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.data]);

  const debouncedGridPosition = useDebouncedValue(gridPosition ?? 1);

  const prediction = useAsync(
    () =>
      predictWinner({
        driver_id: driverId,
        team_id: form.data.team_id,
        circuit_id: entrants?.circuit_id,
        grid_position: debouncedGridPosition,
        round_number: roundNumber,
        mean_previous_grid: form.data.mean_previous_grid,
        mean_previous_position: form.data.mean_previous_position ?? form.data.mean_previous_grid,
        current_driver_wins: form.data.current_driver_wins,
        current_driver_podiums: form.data.current_driver_podiums,
      }),
    [driverId, roundNumber, entrants, form.data, debouncedGridPosition],
    Boolean(form.data && entrants?.circuit_id)
  );

  const gridChanged = form.data && gridPosition !== form.data.grid_position;

  return (
    <div className="page">
      <h1>🔮 Race Predictor</h1>
      <p className="page-intro">
        Pick a real race and driver — we pull their actual recent form and let
        our model call it. Move the grid slot to try a "what if they'd
        started elsewhere".
      </p>

      <div className="controls-row">
        <label>
          Season
          <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>

        <RoundSelect year={year} value={roundNumber} onChange={setRoundNumber} />

        <label>
          Driver
          <select value={driverId} onChange={(e) => setDriverId(e.target.value)}>
            {entrants?.drivers?.map((d) => (
              <option key={d} value={d}>{humanizeSlug(d)}</option>
            ))}
          </select>
        </label>
      </div>

      <AsyncSection loading={form.loading} error={form.error}>
        {form.data && (
          <>
            <div className="form-snapshot">
              <div className="form-stat">
                <span className="form-stat-label">Team</span>
                <span className="form-stat-value">{humanizeSlug(form.data.team_id)}</span>
              </div>
              <div className="form-stat">
                <span className="form-stat-label">Avg. grid, last 3 races</span>
                <span className="form-stat-value">P{form.data.mean_previous_grid.toFixed(1)}</span>
              </div>
              <div className="form-stat">
                <span className="form-stat-label">Avg. finish, last 3 races</span>
                <span className="form-stat-value">
                  {form.data.mean_previous_position != null ? `P${form.data.mean_previous_position.toFixed(1)}` : "—"}
                </span>
              </div>
              <div className="form-stat">
                <span className="form-stat-label">Wins / podiums this season</span>
                <span className="form-stat-value">
                  {form.data.current_driver_wins} / {form.data.current_driver_podiums}
                </span>
              </div>
            </div>

            <div className="controls-grid">
              <label>
                Grid position{gridChanged ? " (hypothetical)" : ""}
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={gridPosition ?? form.data.grid_position}
                  onChange={(e) => setGridPosition(Number(e.target.value))}
                />
                {gridPosition ?? form.data.grid_position}
              </label>
              {gridChanged && (
                <p className="hint">Actually started P{form.data.grid_position}.</p>
              )}
            </div>
          </>
        )}
      </AsyncSection>

      <AsyncSection loading={prediction.loading} error={prediction.error}>
        {prediction.data && (() => {
          const pct = prediction.data.win_probability * 100;
          const outcome = prediction.data.predicted_winner ? "success" : "failure";
          return (
            <>
              <div className={`result ${outcome}`}>
                <div className="result-body">
                  <span>
                    {prediction.data.predicted_winner ? "Expected victory" : "Unlikely to win"}
                  </span>
                  <div className="probability-track">
                    <div className={`probability-fill ${outcome}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
                <span className="result-value">{pct.toFixed(1)}%</span>
              </div>
              {form.data?.actual_position && (
                <p className="hint">
                  What actually happened: finished{" "}
                  {form.data.actual_winner ? "P1 — won the race" : `P${form.data.actual_position}`}.
                </p>
              )}
            </>
          );
        })()}
      </AsyncSection>
    </div>
  );
}
