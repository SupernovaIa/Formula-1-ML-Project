import { useEffect, useState } from "react";
import AsyncSection from "../components/AsyncSection";
import Field from "../components/Field";
import Icon from "../components/Icon";
import PageHead from "../components/PageHead";
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
      <PageHead
        label="XGBoost race-winner model"
        title="Would our model have"
        em="called it?"
        lede={'Pick a real race and driver — we pull their actual recent form and let our model call it. Move the grid slot to try a "what if they\'d started elsewhere".'}
      />

      <div className="a-panel controls">
        <Field label="Season">
          <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </Field>

        <RoundSelect year={year} value={roundNumber} onChange={setRoundNumber} />

        <Field label="Driver">
          <select value={driverId} onChange={(e) => setDriverId(e.target.value)}>
            {entrants?.drivers?.map((d) => (
              <option key={d} value={d}>{humanizeSlug(d)}</option>
            ))}
          </select>
        </Field>
      </div>

      <AsyncSection loading={form.loading} error={form.error}>
        {form.data && (
          <>
            <div className="a-panel form-snapshot">
              <div className="form-stat">
                <span className="a-label">Team</span>
                <span className="form-stat-value">{humanizeSlug(form.data.team_id)}</span>
              </div>
              <div className="form-stat">
                <span className="a-label">Avg. grid, last 3 races</span>
                <span className="form-stat-value">P{form.data.mean_previous_grid.toFixed(1)}</span>
              </div>
              <div className="form-stat">
                <span className="a-label">Avg. finish, last 3 races</span>
                <span className="form-stat-value">
                  {form.data.mean_previous_position != null ? `P${form.data.mean_previous_position.toFixed(1)}` : "—"}
                </span>
              </div>
              <div className="form-stat">
                <span className="a-label">Wins / podiums this season</span>
                <span className="form-stat-value">
                  {form.data.current_driver_wins} / {form.data.current_driver_podiums}
                </span>
              </div>
            </div>

            <div className="a-panel controls">
              <Field label={`Grid position${gridChanged ? " (what if)" : ""}`}>
                <div className="slider-row">
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={gridPosition ?? form.data.grid_position}
                    onChange={(e) => setGridPosition(Number(e.target.value))}
                  />
                  <span className="field-value">P{gridPosition ?? form.data.grid_position}</span>
                </div>
              </Field>
              {gridChanged && (
                <p className="hint">
                  Actually started P{form.data.grid_position}.{" "}
                  <button type="button" className="link-btn" onClick={() => setGridPosition(form.data.grid_position)}>
                    Reset
                  </button>
                </p>
              )}
            </div>
          </>
        )}
      </AsyncSection>

      <AsyncSection loading={prediction.loading} error={prediction.error}>
        {prediction.data && (() => {
          const pct = prediction.data.win_probability * 100;
          const expected = prediction.data.predicted_winner;
          return (
            <>
              {/* The one lit element on the page: the model's call. */}
              <div className="a-seam">
                <div className="prediction">
                  <div className="prediction-body">
                    <p className="prediction-verdict">
                      <Icon name={expected ? "circle-check" : "info"} />
                      {expected ? "Expected victory" : "Unlikely to win"}
                    </p>
                    <div className="prediction-track" role="img" aria-label={`Win probability ${pct.toFixed(1)}%`}>
                      <div className="prediction-fill" style={{ "--p": pct / 100 }} />
                    </div>
                  </div>
                  <span className="prediction-value">{pct.toFixed(1)}%</span>
                </div>
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
