import { useState } from "react";
import AsyncSection from "../components/AsyncSection";
import EChart from "../components/EChart";
import Field from "../components/Field";
import PageHead from "../components/PageHead";
import Segmented from "../components/Segmented";
import { useAsync } from "../hooks/useAsync";
import { multiLineOption } from "../lib/plotlyAdapters";
import { getConstructorsChampionship, getDriversChampionship } from "../api/client";

const YEARS = Array.from({ length: 2024 - 2018 + 1 }, (_, i) => 2018 + i).reverse();
const VIZ_OPTIONS = ["Drivers", "Constructors"];

export default function SeasonReport() {
  const [season, setSeason] = useState(2023);
  const [vizType, setVizType] = useState(VIZ_OPTIONS[0]);

  const drivers = useAsync(() => getDriversChampionship(season, 10), [season], vizType === "Drivers");
  const constructors = useAsync(() => getConstructorsChampionship(season, null), [season], vizType === "Constructors");

  return (
    <div className="page">
      <PageHead title="The title fight," em="race by race" lede="How the drivers' and constructors' battle played out, round by round." />

      <div className="a-panel controls">
        <Field label="Season">
          <select value={season} onChange={(e) => setSeason(Number(e.target.value))}>
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </Field>
        <Segmented options={VIZ_OPTIONS} value={vizType} onChange={setVizType} label="Championship" />
      </div>

      {vizType === "Drivers" && (
        <AsyncSection loading={drivers.loading} error={drivers.error}>
          <EChart option={drivers.data && multiLineOption(drivers.data)} height={520} />
        </AsyncSection>
      )}
      {vizType === "Constructors" && (
        <AsyncSection loading={constructors.loading} error={constructors.error}>
          <EChart option={constructors.data && multiLineOption(constructors.data)} height={520} />
        </AsyncSection>
      )}
    </div>
  );
}
