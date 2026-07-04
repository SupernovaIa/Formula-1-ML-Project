import { useState } from "react";
import AsyncSection from "../components/AsyncSection";
import PlotlyChart from "../components/PlotlyChart";
import { useAsync } from "../hooks/useAsync";
import { humanizeHeader, humanizeSlug } from "../utils/format";
import {
  getDriverPace,
  getFastestLaps,
  getPace,
  getPositionChanges,
  getRaceResults,
  getSeasonRounds,
  getSessionDrivers,
  getStandingsChart,
  getTelemetry,
  getTrack,
  getTyreStrategy,
} from "../api/client";

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i);
const QUALY_VIZ = ["Qualy results", "Pole lap telemetry", "Lap comparison"];
const RACE_VIZ = ["Results", "Position changes", "Driver Pace", "Pace", "Tyre strategies"];

export default function RaceReport() {
  const [year, setYear] = useState(2023);
  const [roundNumber, setRoundNumber] = useState(1);
  const [sessionType, setSessionType] = useState("Qualifying");
  const [loaded, setLoaded] = useState(null);
  const [vizType, setVizType] = useState(QUALY_VIZ[0]);
  const [telemetryMode, setTelemetryMode] = useState("Speed");
  const [selectedDriver, setSelectedDriver] = useState("");
  const [paceKind, setPaceKind] = useState("driver");
  const [threshold, setThreshold] = useState(1.07);
  const [box, setBox] = useState(false);

  const { data: rounds } = useAsync(() => getSeasonRounds(year), [year]);
  const circuitId = rounds?.find((r) => r.round === roundNumber)?.circuit_id;
  const maxRound = rounds?.length ? Math.max(...rounds.map((r) => r.round)) : 1;

  const { data: sessionDrivers } = useAsync(
    () => getSessionDrivers(loaded.year, loaded.round, loaded.sessionType),
    [loaded],
    Boolean(loaded)
  );

  function handleLoad() {
    setLoaded({ year, round: roundNumber, sessionType });
    setVizType(sessionType === "Qualifying" ? QUALY_VIZ[0] : RACE_VIZ[0]);
  }

  return (
    <div className="page">
      <h1>🏁 F1 Grand Prix Dashboard</h1>

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
            onChange={(e) => setRoundNumber(Number(e.target.value))}
          />
        </label>

        {circuitId && <span className="chip">{humanizeSlug(circuitId)}</span>}

        <label>
          Session
          <select value={sessionType} onChange={(e) => setSessionType(e.target.value)}>
            <option value="Qualifying">Qualifying</option>
            <option value="Race">Race</option>
          </select>
        </label>

        <button onClick={handleLoad}>Load Data</button>
      </div>

      {!loaded && <p className="status-text">Pick a season, round and session, then load the data.</p>}

      {loaded && (
        <>
          <div className="controls-row">
            <label>
              Visualization
              <select value={vizType} onChange={(e) => setVizType(e.target.value)}>
                {(loaded.sessionType === "Qualifying" ? QUALY_VIZ : RACE_VIZ).map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>

            {vizType === "Lap comparison" && (
              <label>
                Mode
                <select value={telemetryMode} onChange={(e) => setTelemetryMode(e.target.value)}>
                  <option value="Speed">Speed</option>
                  <option value="Throttle">Throttle</option>
                </select>
              </label>
            )}

            {(vizType === "Driver Pace") && (
              <label>
                Driver
                <select value={selectedDriver} onChange={(e) => setSelectedDriver(e.target.value)}>
                  <option value="">Select a driver</option>
                  {sessionDrivers?.map((d) => (
                    <option key={d.Abbreviation} value={d.Abbreviation}>{d.FullName}</option>
                  ))}
                </select>
              </label>
            )}

            {(vizType === "Driver Pace" || vizType === "Pace") && (
              <label>
                Threshold
                <input
                  type="range"
                  min={1.0}
                  max={2.0}
                  step={0.01}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
                {threshold.toFixed(2)}
              </label>
            )}

            {vizType === "Pace" && (
              <>
                <label>
                  Kind
                  <select value={paceKind} onChange={(e) => setPaceKind(e.target.value)}>
                    <option value="driver">driver</option>
                    <option value="compound">compound</option>
                  </select>
                </label>
                <label>
                  <input type="checkbox" checked={box} onChange={(e) => setBox(e.target.checked)} />
                  Boxplot
                </label>
              </>
            )}
          </div>

          {loaded.sessionType === "Qualifying" && vizType === "Qualy results" && (
            <QualyResults loaded={loaded} />
          )}
          {loaded.sessionType === "Qualifying" && vizType === "Pole lap telemetry" && (
            <PoleLapTelemetry loaded={loaded} />
          )}
          {loaded.sessionType === "Qualifying" && vizType === "Lap comparison" && (
            <LapComparison loaded={loaded} mode={telemetryMode} drivers={sessionDrivers} />
          )}

          {loaded.sessionType === "Race" && vizType === "Results" && <RaceResults loaded={loaded} />}
          {loaded.sessionType === "Race" && vizType === "Position changes" && (
            <SingleChart fetcher={() => getPositionChanges(loaded.year, loaded.round, loaded.sessionType)} deps={[loaded]} />
          )}
          {loaded.sessionType === "Race" && vizType === "Driver Pace" && selectedDriver && (
            <SingleChart
              fetcher={() => getDriverPace(loaded.year, loaded.round, loaded.sessionType, selectedDriver, threshold)}
              deps={[loaded, selectedDriver, threshold]}
            />
          )}
          {loaded.sessionType === "Race" && vizType === "Pace" && (
            <SingleChart
              fetcher={() => getPace(loaded.year, loaded.round, loaded.sessionType, paceKind, threshold, box)}
              deps={[loaded, paceKind, threshold, box]}
            />
          )}
          {loaded.sessionType === "Race" && vizType === "Tyre strategies" && (
            <SingleChart fetcher={() => getTyreStrategy(loaded.year, loaded.round, loaded.sessionType)} deps={[loaded]} />
          )}
        </>
      )}
    </div>
  );
}

function SingleChart({ fetcher, deps }) {
  const { data, loading, error } = useAsync(fetcher, deps);
  return (
    <AsyncSection loading={loading} error={error}>
      <PlotlyChart figure={data} />
    </AsyncSection>
  );
}

function ResultsTable({ rows }) {
  if (!rows?.length) return null;
  const columns = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{humanizeHeader(c)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{columns.map((c) => <td key={c}>{String(row[c] ?? "")}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QualyResults({ loaded }) {
  const results = useAsync(() => getRaceResults(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  const chart = useAsync(() => getFastestLaps(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  return (
    <>
      <AsyncSection loading={results.loading} error={results.error}>
        <ResultsTable rows={results.data} />
      </AsyncSection>
      <AsyncSection loading={chart.loading} error={chart.error}>
        <PlotlyChart figure={chart.data} />
      </AsyncSection>
    </>
  );
}

function PoleLapTelemetry({ loaded }) {
  const track = useAsync(() => getTrack(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  const telemetry = useAsync(() => getTelemetry(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  return (
    <>
      <AsyncSection loading={track.loading} error={track.error}>
        <PlotlyChart figure={track.data} />
      </AsyncSection>
      <AsyncSection loading={telemetry.loading} error={telemetry.error}>
        <PlotlyChart figure={telemetry.data} />
      </AsyncSection>
    </>
  );
}

function LapComparison({ loaded, mode, drivers }) {
  const abbreviations = drivers?.map((d) => d.Abbreviation) ?? [];
  const { data, loading, error } = useAsync(
    () => getTelemetry(loaded.year, loaded.round, loaded.sessionType, mode, abbreviations),
    [loaded, mode, drivers]
  );
  return (
    <AsyncSection loading={loading} error={error}>
      <PlotlyChart figure={data} />
    </AsyncSection>
  );
}

function RaceResults({ loaded }) {
  const results = useAsync(() => getRaceResults(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  const chart = useAsync(() => getStandingsChart(loaded.year, loaded.round, loaded.sessionType), [loaded]);
  return (
    <>
      <AsyncSection loading={results.loading} error={results.error}>
        <ResultsTable rows={results.data} />
      </AsyncSection>
      <AsyncSection loading={chart.loading} error={chart.error}>
        <PlotlyChart figure={chart.data} />
      </AsyncSection>
    </>
  );
}
