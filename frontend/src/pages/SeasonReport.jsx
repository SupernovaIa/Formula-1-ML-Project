import { useState } from "react";
import AsyncSection from "../components/AsyncSection";
import EChart from "../components/EChart";
import { useAsync } from "../hooks/useAsync";
import { multiLineOption } from "../lib/plotlyAdapters";
import { getConstructorsChampionship, getDriversChampionship } from "../api/client";

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i).reverse();
const VIZ_OPTIONS = ["Drivers", "Constructors"];

export default function SeasonReport() {
  const [season, setSeason] = useState(2023);
  const [loadedSeason, setLoadedSeason] = useState(null);
  const [vizType, setVizType] = useState(VIZ_OPTIONS[0]);

  const drivers = useAsync(
    () => getDriversChampionship(loadedSeason, 10),
    [loadedSeason],
    Boolean(loadedSeason) && vizType === "Drivers"
  );
  const constructors = useAsync(
    () => getConstructorsChampionship(loadedSeason, null),
    [loadedSeason],
    Boolean(loadedSeason) && vizType === "Constructors"
  );

  return (
    <div className="page">
      <h1>🏆 Championship</h1>
      {!loadedSeason && <p className="page-intro">See how the title battle played out, race by race.</p>}

      <div className="controls-row">
        <label>
          Season
          <select value={season} onChange={(e) => setSeason(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </label>
        <button onClick={() => setLoadedSeason(season)}>Load</button>
      </div>

      {loadedSeason && (
        <>
          <div className="controls-row">
            <div className="segment-group" role="tablist" aria-label="Championship">
              {VIZ_OPTIONS.map((v) => (
                <button
                  key={v}
                  type="button"
                  role="tab"
                  aria-selected={vizType === v}
                  className={`segment-btn ${vizType === v ? "active" : ""}`}
                  onClick={() => setVizType(v)}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          {vizType === "Drivers" && (
            <AsyncSection loading={drivers.loading} error={drivers.error}>
              <EChart option={drivers.data && multiLineOption(drivers.data)} />
            </AsyncSection>
          )}
          {vizType === "Constructors" && (
            <AsyncSection loading={constructors.loading} error={constructors.error}>
              <EChart option={constructors.data && multiLineOption(constructors.data)} />
            </AsyncSection>
          )}
        </>
      )}
    </div>
  );
}
