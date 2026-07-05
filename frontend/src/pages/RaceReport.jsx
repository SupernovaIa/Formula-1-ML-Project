import { useState } from "react";
import AsyncSection from "../components/AsyncSection";
import EChart from "../components/EChart";
import RoundSelect from "../components/RoundSelect";
import { useAsync } from "../hooks/useAsync";
import { humanizeHeader, humanizeSlug } from "../utils/format";
import {
  barOption,
  multiLineOption,
  paceBoxplotOption,
  scatterGroupsOption,
  telemetryOption,
  trackOption,
  tyreStrategyOption,
} from "../lib/plotlyAdapters";
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

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i).reverse();
const QUALY_VIZ = ["Results", "Pole lap analysis", "Compare fastest laps"];
const RACE_VIZ = ["Results", "How the race unfolded", "Driver pace", "Pace comparison", "Tyre strategy"];
const OUTLIER_THRESHOLD = 1.07; // F1's own "107% rule" for slow-lap cutoffs
const NO_THRESHOLD = 3.0; // effectively unfiltered

export default function RaceReport() {
  const [year, setYear] = useState(2023);
  const [roundNumber, setRoundNumber] = useState(1);
  const [sessionType, setSessionType] = useState("Qualifying");
  const [loaded, setLoaded] = useState(null);
  const [vizType, setVizType] = useState(QUALY_VIZ[0]);
  const [telemetryMode, setTelemetryMode] = useState("Speed");
  const [selectedDriver, setSelectedDriver] = useState("");
  const [paceKind, setPaceKind] = useState("driver");
  const [excludeOutliers, setExcludeOutliers] = useState(true);
  const [showPoints, setShowPoints] = useState(false);

  const threshold = excludeOutliers ? OUTLIER_THRESHOLD : NO_THRESHOLD;

  const { data: rounds } = useAsync(() => getSeasonRounds(year), [year]);
  const circuitId = rounds?.find((r) => r.round === roundNumber)?.circuit_id;

  const { data: sessionDrivers } = useAsync(
    () => getSessionDrivers(loaded.year, loaded.round, loaded.sessionType),
    [loaded],
    Boolean(loaded)
  );

  function handleLoad() {
    setLoaded({ year, round: roundNumber, sessionType, circuitId });
    setVizType(sessionType === "Qualifying" ? QUALY_VIZ[0] : RACE_VIZ[0]);
  }

  return (
    <div className="page">
      <h1>{loaded ? `🏁 ${humanizeSlug(loaded.circuitId)} — Round ${loaded.round}, ${loaded.year}` : "🏁 Race Weekend"}</h1>
      {!loaded && <p className="page-intro">Pick a season and race to explore its qualifying and race sessions.</p>}

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
          Session
          <select value={sessionType} onChange={(e) => setSessionType(e.target.value)}>
            <option value="Qualifying">Qualifying</option>
            <option value="Race">Race</option>
          </select>
        </label>

        <button onClick={handleLoad}>Load</button>
      </div>

      {loaded && (
        <>
          <div className="controls-row">
            <label>
              View
              <select value={vizType} onChange={(e) => setVizType(e.target.value)}>
                {(loaded.sessionType === "Qualifying" ? QUALY_VIZ : RACE_VIZ).map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>

            {vizType === "Compare fastest laps" && (
              <label>
                Mode
                <select value={telemetryMode} onChange={(e) => setTelemetryMode(e.target.value)}>
                  <option value="Speed">Speed</option>
                  <option value="Throttle">Throttle</option>
                </select>
              </label>
            )}

            {(vizType === "Driver pace") && (
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

            {(vizType === "Driver pace" || vizType === "Pace comparison") && (
              <label>
                <input
                  type="checkbox"
                  checked={excludeOutliers}
                  onChange={(e) => setExcludeOutliers(e.target.checked)}
                />
                Exclude outlier laps (pit stops, safety car)
              </label>
            )}

            {vizType === "Pace comparison" && (
              <>
                <label>
                  Group by
                  <select value={paceKind} onChange={(e) => setPaceKind(e.target.value)}>
                    <option value="driver">Driver</option>
                    <option value="compound">Tyre compound</option>
                  </select>
                </label>
                <label>
                  <input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} />
                  Show individual laps
                </label>
              </>
            )}
          </div>

          {loaded.sessionType === "Qualifying" && vizType === "Results" && (
            <QualyResults loaded={loaded} />
          )}
          {loaded.sessionType === "Qualifying" && vizType === "Pole lap analysis" && (
            <PoleLapTelemetry loaded={loaded} />
          )}
          {loaded.sessionType === "Qualifying" && vizType === "Compare fastest laps" && (
            <LapComparison loaded={loaded} mode={telemetryMode} drivers={sessionDrivers} />
          )}

          {loaded.sessionType === "Race" && vizType === "Results" && <RaceResults loaded={loaded} />}
          {loaded.sessionType === "Race" && vizType === "How the race unfolded" && (
            <SingleChart
              fetcher={() => getPositionChanges(loaded.year, loaded.round, loaded.sessionType)}
              deps={[loaded]}
              adapter={multiLineOption}
            />
          )}
          {loaded.sessionType === "Race" && vizType === "Driver pace" && selectedDriver && (
            <SingleChart
              fetcher={() => getDriverPace(loaded.year, loaded.round, loaded.sessionType, selectedDriver, threshold)}
              deps={[loaded, selectedDriver, threshold]}
              adapter={scatterGroupsOption}
            />
          )}
          {loaded.sessionType === "Race" && vizType === "Pace comparison" && (
            <SingleChart
              fetcher={() => getPace(loaded.year, loaded.round, loaded.sessionType, paceKind, threshold)}
              deps={[loaded, paceKind, threshold]}
              adapter={(data) => paceBoxplotOption(data, { showPoints })}
            />
          )}
          {loaded.sessionType === "Race" && vizType === "Tyre strategy" && (
            <SingleChart
              fetcher={() => getTyreStrategy(loaded.year, loaded.round, loaded.sessionType)}
              deps={[loaded]}
              adapter={tyreStrategyOption}
            />
          )}
        </>
      )}
    </div>
  );
}

function SingleChart({ fetcher, deps, adapter }) {
  const { data, loading, error } = useAsync(fetcher, deps);
  return (
    <AsyncSection loading={loading} error={error}>
      <EChart option={data && adapter(data)} />
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
        <EChart option={chart.data && barOption(chart.data)} />
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
        <EChart option={track.data && trackOption(track.data)} />
      </AsyncSection>
      <AsyncSection loading={telemetry.loading} error={telemetry.error}>
        <EChart option={telemetry.data && telemetryOption(telemetry.data)} />
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
      <EChart option={data && telemetryOption(data)} />
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
        <EChart option={chart.data && barOption(chart.data)} />
      </AsyncSection>
    </>
  );
}
