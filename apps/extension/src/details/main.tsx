import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { DetailsApp } from "./DetailsApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DetailsApp />
  </StrictMode>,
);
