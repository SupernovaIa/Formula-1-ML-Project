import { useEffect, useState } from "react";
import AsyncSection from "../components/AsyncSection";
import { useAsync } from "../hooks/useAsync";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { getRoundEntrants, getSeasonRounds, predictWinner } from "../api/client";
import { humanizeSlug } from "../utils/format";

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i);

export default function WinnerPrediction() {
  const [year, setYear] = useState(2023);
  const [roundNumber, setRoundNumber] = useState(1);
  const [driverId, setDriverId] = useState("");
  const [teamId, setTeamId] = useState("");
  const [meanPreviousGrid, setMeanPreviousGrid] = useState(5);
  const [meanPreviousPosition, setMeanPreviousPosition] = useState(5);
  const [gridPosition, setGridPosition] = useState(5);
  const [currentDriverWins, setCurrentDriverWins] = useState(0);
  const [currentDriverPodiums, setCurrentDriverPodiums] = useState(0);

  const { data: rounds } = useAsync(() => getSeasonRounds(year), [year]);
  const maxRound = rounds?.length ? Math.max(...rounds.map((r) => r.round)) : 1;

  const { data: entrants } = useAsync(() => getRoundEntrants(year, roundNumber), [year, roundNumber]);

  useEffect(() => {
    if (entrants?.drivers?.length && !entrants.drivers.includes(driverId)) {
      setDriverId(entrants.drivers[0]);
    }
    if (entrants?.teams?.length && !entrants.teams.includes(teamId)) {
      setTeamId(entrants.teams[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entrants]);

  useEffect(() => {
    if (roundNumber === 1) {
      setCurrentDriverWins(0);
      setCurrentDriverPodiums(0);
    } else if (currentDriverWins === roundNumber - 1) {
      setCurrentDriverPodiums(currentDriverWins);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roundNumber, currentDriverWins]);

  // Debounce the continuous inputs (sliders/number fields) so dragging doesn't
  // fire a request per pixel — the UI itself stays responsive, only the API
  // call waits for things to settle.
  const debouncedGridPosition = useDebouncedValue(gridPosition);
  const debouncedMeanPreviousGrid = useDebouncedValue(meanPreviousGrid);
  const debouncedMeanPreviousPosition = useDebouncedValue(meanPreviousPosition);
  const debouncedCurrentDriverWins = useDebouncedValue(currentDriverWins);
  const debouncedCurrentDriverPodiums = useDebouncedValue(currentDriverPodiums);

  const prediction = useAsync(
    () =>
      predictWinner({
        driver_id: driverId,
        team_id: teamId,
        circuit_id: entrants?.circuit_id,
        grid_position: debouncedGridPosition,
        round_number: roundNumber,
        mean_previous_grid: debouncedMeanPreviousGrid,
        mean_previous_position: debouncedMeanPreviousPosition,
        current_driver_wins: debouncedCurrentDriverWins,
        current_driver_podiums: debouncedCurrentDriverPodiums,
      }),
    [
      driverId,
      teamId,
      entrants,
      debouncedGridPosition,
      roundNumber,
      debouncedMeanPreviousGrid,
      debouncedMeanPreviousPosition,
      debouncedCurrentDriverWins,
      debouncedCurrentDriverPodiums,
    ],
    Boolean(driverId && teamId && entrants?.circuit_id)
  );

  return (
    <div className="page">
      <h1>🏠 Race winner prediction using ML 🔮</h1>
      <p>Use this app to predict future 🚀</p>

      <div className="controls-row">
        <label>
          Season
          <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>
        <label>
          Round
          <input
            type="number"
            min={1}
            max={maxRound}
            value={roundNumber}
            onChange={(e) => setRoundNumber(Math.max(1, Number(e.target.value) || 1))}
          />
        </label>
        {entrants?.circuit_id && <span className="chip">{humanizeSlug(entrants.circuit_id)}</span>}
      </div>

      <h2>🔧 Features</h2>
      <div className="controls-grid">
        <label>
          Driver
          <select value={driverId} onChange={(e) => setDriverId(e.target.value)}>
            {entrants?.drivers?.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>

        <label>
          Team
          <select value={teamId} onChange={(e) => setTeamId(e.target.value)}>
            {entrants?.teams?.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>

        <label>
          Previous grid position (mean, last 3 races)
          <input
            type="number"
            min={1}
            max={20}
            value={meanPreviousGrid}
            onChange={(e) => setMeanPreviousGrid(Number(e.target.value))}
          />
        </label>

        <label>
          Previous position (mean, last 3 races)
          <input
            type="number"
            min={1}
            max={20}
            value={meanPreviousPosition}
            onChange={(e) => setMeanPreviousPosition(Number(e.target.value))}
          />
        </label>

        <label>
          Grid position
          <input
            type="range"
            min={1}
            max={20}
            value={gridPosition}
            onChange={(e) => setGridPosition(Number(e.target.value))}
          />
          {gridPosition}
        </label>

        {roundNumber === 1 ? (
          <p className="hint">Current driver wins and podiums are set to 0 in round 1.</p>
        ) : (
          <>
            <label>
              Current wins
              <input
                type="range"
                min={0}
                max={roundNumber - 1}
                value={currentDriverWins}
                onChange={(e) => setCurrentDriverWins(Number(e.target.value))}
              />
              {currentDriverWins}
            </label>

            {currentDriverWins === roundNumber - 1 ? (
              <p className="hint">Current driver podiums is set to {currentDriverPodiums}.</p>
            ) : (
              <label>
                Current podiums
                <input
                  type="range"
                  min={currentDriverWins}
                  max={roundNumber - 1}
                  value={currentDriverPodiums}
                  onChange={(e) => setCurrentDriverPodiums(Number(e.target.value))}
                />
                {currentDriverPodiums}
              </label>
            )}
          </>
        )}
      </div>

      <AsyncSection loading={prediction.loading} error={prediction.error}>
        {prediction.data && (() => {
          const pct = prediction.data.win_probability * 100;
          const outcome = prediction.data.predicted_winner ? "success" : "failure";
          return (
            <div className={`result ${outcome}`}>
              <div className="result-body">
                <span>
                  {prediction.data.predicted_winner
                    ? "Expected victory"
                    : "Unlikely to win"}
                </span>
                <div className="probability-track">
                  <div className={`probability-fill ${outcome}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
              <span className="result-value">{pct.toFixed(1)}%</span>
            </div>
          );
        })()}
      </AsyncSection>
    </div>
  );
}
