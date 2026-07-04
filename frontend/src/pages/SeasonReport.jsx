import { useState } from "react";
import AsyncSection from "../components/AsyncSection";
import PlotlyChart from "../components/PlotlyChart";
import { useAsync } from "../hooks/useAsync";
import { getConstructorsChampionship, getDriversChampionship } from "../api/client";

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i);
const VIZ_OPTIONS = ["Drivers championship", "Constructors championship"];

export default function SeasonReport() {
  const [season, setSeason] = useState(2023);
  const [loadedSeason, setLoadedSeason] = useState(null);
  const [vizType, setVizType] = useState(VIZ_OPTIONS[0]);

  const drivers = useAsync(
    () => getDriversChampionship(loadedSeason, 10),
    [loadedSeason],
    Boolean(loadedSeason) && vizType === "Drivers championship"
  );
  const constructors = useAsync(
    () => getConstructorsChampionship(loadedSeason, null),
    [loadedSeason],
    Boolean(loadedSeason) && vizType === "Constructors championship"
  );

  return (
    <div className="page">
      <h1>🏁 F1 Season Dashboard</h1>

      <div className="controls-row">
        <label>
          Season
          <select value={season} onChange={(e) => setSeason(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>
        <button onClick={() => setLoadedSeason(season)}>Load Data</button>
      </div>

      {!loadedSeason && <p className="status-text">Pick a season, then load the data.</p>}

      {loadedSeason && (
        <>
          <div className="controls-row">
            <label>
              Visualization
              <select value={vizType} onChange={(e) => setVizType(e.target.value)}>
                {VIZ_OPTIONS.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </label>
          </div>

          {vizType === "Drivers championship" && (
            <AsyncSection loading={drivers.loading} error={drivers.error}>
              <PlotlyChart figure={drivers.data} />
            </AsyncSection>
          )}
          {vizType === "Constructors championship" && (
            <AsyncSection loading={constructors.loading} error={constructors.error}>
              <PlotlyChart figure={constructors.data} />
            </AsyncSection>
          )}
        </>
      )}
    </div>
  );
}
