import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import RaceReport from "./pages/RaceReport";
import SeasonReport from "./pages/SeasonReport";
import CircuitClustering from "./pages/CircuitClustering";
import WinnerPrediction from "./pages/WinnerPrediction";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="/race-report" element={<RaceReport />} />
          <Route path="/season-report" element={<SeasonReport />} />
          <Route path="/circuit-clustering" element={<CircuitClustering />} />
          <Route path="/winner-prediction" element={<WinnerPrediction />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
