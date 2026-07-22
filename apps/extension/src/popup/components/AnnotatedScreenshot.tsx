import type { AnnotationBox } from "../../shared/annotations";
import { annotationColor } from "../../shared/annotations";
import { useMemo, useRef, useState } from "react";

interface Props {
  imageUrl: string | null | undefined;
  boxes: AnnotationBox[];
}

export function AnnotatedScreenshot({ imageUrl, boxes }: Props) {
  const [zoom, setZoom] = useState(1);
  const wrapRef = useRef<HTMLDivElement>(null);
  const drawable = useMemo(() => boxes.filter((b) => b.width > 0 && b.height > 0), [boxes]);

  if (!imageUrl) {
    return <p className="meta">No screenshot available for this scan.</p>;
  }

  return (
    <div className="annotator">
      <div className="annotator-toolbar">
        <button type="button" className="btn btn-secondary" onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))} aria-label="Zoom out">
          −
        </button>
        <span className="meta">{Math.round(zoom * 100)}%</span>
        <button type="button" className="btn btn-secondary" onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))} aria-label="Zoom in">
          +
        </button>
      </div>
      <div className="annotator-viewport" ref={wrapRef}>
        <div className="annotator-stage" style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}>
          <img src={imageUrl} alt="Page screenshot with consent annotations" />
          {drawable.map((box) => (
            <div
              key={box.id}
              className="annot-box"
              style={{
                left: box.x,
                top: box.y,
                width: box.width,
                height: box.height,
                borderColor: annotationColor(box.kind),
              }}
              title={`${box.label} (${Math.round(box.confidence * 100)}%)`}
            >
              <span style={{ background: annotationColor(box.kind) }}>
                {box.label} · {Math.round(box.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
      <ul className="annot-legend">
        {(["banner", "accept", "reject", "settings", "overlay", "checkbox"] as const).map((k) => (
          <li key={k}>
            <i style={{ background: annotationColor(k) }} /> {k}
          </li>
        ))}
      </ul>
    </div>
  );
}
