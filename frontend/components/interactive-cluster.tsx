import React, { useId, useMemo, useState } from "react";

type InstanceType = {
    id: string;
    name: string;
    vcpu: number;
    dramGb: number;
    bandwidthGbps: number; // interpreted as per-node bandwidth
};

type ClusterConfig = {
    nodes: number;
    instanceTypeId: string;
};

type NodeViz = {
    id: string;
    index: number;
    cx: number;
    cy: number;
    x: number;
    y: number;
    w: number;
    h: number;
};

type LinkViz = {
    id: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    strokeWidth: number;
    stroke: string;
    opacity: number;
    label?: { text: string; x: number; y: number };
};

type ClusterViz = {
    nodes: NodeViz[];
    links: LinkViz[];
    s3Box: { x: number; y: number; w: number; h: number };
    totals: { totalVcpu: number; totalDramGb: number; totalBandwidthGbps: number };
    warnings: string[];
};

function clamp(n: number, min: number, max: number) {
    return Math.max(min, Math.min(max, n));
}

function bandwidthToStrokeWidth(bwGbps: number) {
    // 1..400 -> ~1..8-ish
    const safe = Math.max(1, bwGbps);
    return clamp(Math.log2(safe) * 0.9, 1, 8);
}

function computeClusterViz(config: ClusterConfig, it: InstanceType): ClusterViz {
    const nodesN = clamp(config.nodes, 1, 256);

    const totalVcpu = nodesN * it.vcpu;
    const totalDramGb = nodesN * it.dramGb;
    const totalBandwidthGbps = nodesN * it.bandwidthGbps;

    const warnings: string[] = [];
    if (it.bandwidthGbps < 5) warnings.push("Low network bandwidth per node");
    if (nodesN > 160) warnings.push("High node count: inter-node links are thinned for readability/perf");

    // SVG canvas
    const SVG_W = 980;
    const SVG_H = 520;

    // Reserve space for S3 box
    const s3Gap = 18;
    const s3H = 54;

    // Cluster layout area
    const marginX = 50;
    const top = 26;
    const bottom = 20;
    const availW = SVG_W - marginX * 2;
    const availH = SVG_H - top - bottom - s3Gap - s3H;

    // Deterministic grid (stable)
    const cols = Math.ceil(Math.sqrt(nodesN));
    const rows = Math.ceil(nodesN / cols);

    const pad = 14;
    const rawW = (availW - (cols - 1) * pad) / cols;
    const rawH = (availH - (rows - 1) * pad) / rows;

    // Node size (big enough to show some vCPU squares + DRAM bar)
    const nodeW = clamp(rawW, 56, 150);
    const nodeH = clamp(rawH, 70, 140);

    const gridW = cols * nodeW + (cols - 1) * pad;
    const gridH = rows * nodeH + (rows - 1) * pad;
    const startX = (SVG_W - gridW) / 2;
    const startY = top + (availH - gridH) / 2;

    const nodeVizes: NodeViz[] = [];
    for (let i = 0; i < nodesN; i++) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const x = startX + c * (nodeW + pad);
        const y = startY + r * (nodeH + pad);
        nodeVizes.push({
            id: `node-${i}`,
            index: i,
            x,
            y,
            w: nodeW,
            h: nodeH,
            cx: x + nodeW / 2,
            cy: y + nodeH / 2,
        });
    }

    // S3 box under the grid
    const maxY = nodeVizes.reduce((m, n) => Math.max(m, n.y + n.h), 0);
    const s3Box = {
        x: 70,
        y: Math.min(SVG_H - bottom - s3H, maxY + s3Gap),
        w: SVG_W - 140,
        h: s3H,
    };

    // Bandwidth visual encoding on links (from instance type)
    const bwStroke = bandwidthToStrokeWidth(it.bandwidthGbps);
    const bwNorm = clamp(it.bandwidthGbps / 400, 0, 1);
    const meshStroke = clamp(bwStroke * 0.35, 1, 4);
    const meshOpacity = 0.04 + bwNorm * 0.10;

    // Inter-node links
    const links: LinkViz[] = [];

    // Label rules (avoid clutter):
    // - For small clusters: label many neighbor links
    // - For larger clusters: label a sparse subset
    const bwLabel = `${it.bandwidthGbps} Gbps`;
    const shouldLabel = (edgeIndex: number, nodesCount: number) => {
        if (nodesCount <= 20) return true;
        if (nodesCount <= 64) return edgeIndex % 2 === 0;
        return edgeIndex % 6 === 0;
    };

    // Full mesh up to 26 (labels on all would still be too much; we label only some)
    if (nodesN <= 26) {
        let e = 0;
        for (let i = 0; i < nodeVizes.length; i++) {
            for (let j = i + 1; j < nodeVizes.length; j++) {
                const a = nodeVizes[i];
                const b = nodeVizes[j];
                const midX = (a.cx + b.cx) / 2;
                const midY = (a.cy + b.cy) / 2;
                links.push({
                    id: `m-${i}-${j}`,
                    x1: a.cx,
                    y1: a.cy,
                    x2: b.cx,
                    y2: b.cy,
                    strokeWidth: meshStroke,
                    stroke: "#2b2f36",
                    opacity: meshOpacity,
                    label: shouldLabel(e++, nodesN) ? { text: bwLabel, x: midX, y: midY } : undefined,
                });
            }
        }
    } else {
        // Thinned mesh: neighbors + a couple longer links
        const seen = new Set<string>();
        let e = 0;

        const add = (i: number, j: number, wantLabel: boolean) => {
            if (j < 0 || j >= nodeVizes.length || i === j) return;
            const a = nodeVizes[i];
            const b = nodeVizes[j];
            const key = i < j ? `m-${i}-${j}` : `m-${j}-${i}`;
            if (seen.has(key)) return;
            seen.add(key);
            const midX = (a.cx + b.cx) / 2;
            const midY = (a.cy + b.cy) / 2;
            links.push({
                id: key,
                x1: a.cx,
                y1: a.cy,
                x2: b.cx,
                y2: b.cy,
                strokeWidth: meshStroke,
                stroke: "#2b2f36",
                opacity: meshOpacity,
                label: wantLabel && shouldLabel(e++, nodesN) ? { text: bwLabel, x: midX, y: midY } : undefined,
            });
        };

        for (let i = 0; i < nodeVizes.length; i++) {
            // grid neighbors (these are the most readable places for labels)
            add(i, i + 1, true);
            add(i, i + cols, true);
            add(i, i - 1, false);
            add(i, i - cols, false);
            add(i, i + cols + 1, false);
            add(i, i + cols - 1, false);

            // longer links (rarely labeled)
            add(i, (i + Math.floor(nodeVizes.length / 3)) % nodeVizes.length, false);
            add(i, (i + Math.floor(nodeVizes.length / 2)) % nodeVizes.length, false);
        }
    }

    return {
        nodes: nodeVizes,
        links,
        s3Box,
        totals: { totalVcpu, totalDramGb, totalBandwidthGbps },
        warnings,
    };
}

function formatNumber(n: number) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function Badge({ children }: { children: React.ReactNode }) {
    return (
        <span
            style={{
                display: "inline-flex",
                alignItems: "center",
                padding: "4px 10px",
                borderRadius: 999,
                border: "1px solid rgba(0,0,0,0.12)",
                background: "rgba(255,255,255,0.7)",
                fontSize: 12,
                gap: 8,
            }}
        >
            {children}
        </span>
    );
}

function Slider({
    label,
    value,
    min,
    max,
    step,
    onChange,
    testId,
}: {
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    onChange: (v: number) => void;
    testId: string;
}) {
    const id = useId();
    return (
        <label htmlFor={id} style={{ display: "grid", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <span style={{ fontSize: 13, color: "#111" }}>{label}</span>
                <Badge>
                    <span style={{ fontVariantNumeric: "tabular-nums" }}>{formatNumber(value)}</span>
                </Badge>
            </div>
            <input
                id={id}
                data-testid={testId}
                type="range"
                min={min}
                max={max}
                step={step ?? 1}
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
            />
        </label>
    );
}

function TextPill({ x, y, text }: { x: number; y: number; text: string }) {
    // Approximate text width (good enough for simple labels)
    const fontSize = 10;
    const padX = 6;
    const padY = 3;
    const estW = text.length * (fontSize * 0.55) + padX * 2;
    const estH = fontSize + padY * 2;

    return (
        <g>
            <rect
                x={x - estW / 2}
                y={y - estH / 2}
                width={estW}
                height={estH}
                rx={7}
                fill="#ffffff"
                opacity={0.85}
                stroke="#000"
                strokeOpacity={0.08}
            />
            <text x={x} y={y + fontSize * 0.35} textAnchor="middle" fontSize={fontSize} fill="#222" opacity={0.9}>
                {text}
            </text>
        </g>
    );
}

function VcpuSquares({
    x,
    y,
    w,
    h,
    vcpu,
    dramReserve,
}: {
    x: number;
    y: number;
    w: number;
    h: number;
    vcpu: number;
    dramReserve: number;
}) {
    // Layout rules:
    // - Wrap after 8 boxes per row
    // - Box size adapts based on number of vCPUs and available space
    // - Render all boxes if they fit; otherwise show what fits + a "+N" box

    const margin = 7;
    const gap = 5;

    const labelReserve = 16; // node label area
    const availW = Math.max(0, w - margin * 2);
    const availH = Math.max(0, h - margin * 2 - labelReserve - dramReserve);

    const cols = clamp(Math.min(8, vcpu), 1, 8);
    const rowsNeeded = Math.ceil(vcpu / cols);

    // Allow larger boxes for small vcpu counts
    const maxBox = 34;
    const minBox = 7;

    const boxByW = (availW - (cols - 1) * gap) / cols;
    const boxByH = (availH - (rowsNeeded - 1) * gap) / Math.max(1, rowsNeeded);
    let box = Math.floor(Math.min(maxBox, boxByW, boxByH));
    box = clamp(box, minBox, maxBox);

    // Determine how many actually fit at this box size
    const fitCols = clamp(Math.floor((availW + gap) / (box + gap)), 1, 8);
    const fitRows = clamp(Math.floor((availH + gap) / (box + gap)), 1, 99);
    const maxFit = fitCols * fitRows;

    const left = x + margin;
    const top = y + margin;

    const makeBox = (bx: number, by: number, key: string, text: string) => (
        <g key={key}>
            <rect x={bx} y={by} width={box} height={box} rx={5} fill="#ffffff" stroke="#111111" strokeOpacity={0.22} />
            <text
                x={bx + box / 2}
                y={by + box / 2 + 2.5}
                textAnchor="middle"
                fontSize={clamp(box * 0.34, 6, 12)}
                fill="#222"
            >
                {text}
            </text>
        </g>
    );

    // If the node is extremely small, render one summary box
    if (maxFit <= 1) {
        const txt = vcpu >= 1000 ? `+${Math.floor(vcpu / 1000)}k` : `+${vcpu}`;
        return <g>{makeBox(left, top, "v-sum", vcpu === 1 ? "vCPU" : txt)}</g>;
    }

    const renderCount = vcpu <= maxFit ? vcpu : Math.max(0, maxFit - 1);
    const remaining = vcpu - renderCount;

    // Center the grid within the available area
    const usedCols = Math.min(fitCols, renderCount + (remaining > 0 ? 1 : 0));
    const usedRows = Math.ceil((renderCount + (remaining > 0 ? 1 : 0)) / fitCols);
    const gridW = usedCols * box + Math.max(0, usedCols - 1) * gap;
    const gridH = usedRows * box + Math.max(0, usedRows - 1) * gap;

    const offsetX = Math.max(0, (availW - gridW) / 2);
    const offsetY = Math.max(0, (availH - gridH) / 2);

    const boxes: React.ReactNode[] = [];
    for (let i = 0; i < renderCount; i++) {
        const r = Math.floor(i / fitCols);
        const c = i % fitCols;
        const bx = left + offsetX + c * (box + gap);
        const by = top + offsetY + r * (box + gap);
        boxes.push(makeBox(bx, by, `v-${i}`, "vCPU"));
    }

    if (remaining > 0) {
        const i = renderCount;
        const r = Math.floor(i / fitCols);
        const c = i % fitCols;
        const bx = left + offsetX + c * (box + gap);
        const by = top + offsetY + r * (box + gap);
        const txt = remaining >= 1000 ? `+${Math.floor(remaining / 1000)}k` : `+${remaining}`;
        boxes.push(makeBox(bx, by, "v-more", txt));
    }

    return <g>{boxes}</g>;
}

function DramBar({ x, y, w, h, text }: { x: number; y: number; w: number; h: number; text: string }) {
    return (
        <g>
            <rect x={x} y={y} width={w} height={h} rx={6} fill="#ffffff" stroke="#111111" strokeOpacity={0.18} />
            <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={11} fill="#222" opacity={0.9}>
                {text}
            </text>
        </g>
    );
}

export default function InteractiveCluster() {
    const [instanceTypes, setInstanceTypes] = useState<InstanceType[]>([
        { id: "1", name: "Instance 1", vcpu: 4, dramGb: 16, bandwidthGbps: 25 },
        { id: "2", name: "Instance 2", vcpu: 8, dramGb: 32, bandwidthGbps: 50 },
        { id: "3", name: "Instance 3", vcpu: 16, dramGb: 64, bandwidthGbps: 100 },
    ]);

    const [config, setConfig] = useState<ClusterConfig>({ nodes: 16, instanceTypeId: "3" });

    const selected = useMemo(() => {
        return instanceTypes.find((t) => t.id === config.instanceTypeId) ?? instanceTypes[0];
    }, [instanceTypes, config.instanceTypeId]);

    // Edit instance UI
    const [isEditing, setIsEditing] = useState(false);
    const [draft, setDraft] = useState(() => ({
        name: selected.name,
        vcpu: selected.vcpu,
        dramGb: selected.dramGb,
        bandwidthGbps: selected.bandwidthGbps,
    }));

    // Keep draft in sync when switching selected instance (unless user is actively editing)
    React.useEffect(() => {
        if (!isEditing) {
            setDraft({
                name: selected.name,
                vcpu: selected.vcpu,
                dramGb: selected.dramGb,
                bandwidthGbps: selected.bandwidthGbps,
            });
        }
    }, [selected.id, isEditing]);

    const viz = useMemo(() => computeClusterViz(config, selected), [config, selected]);

    // vCPU -> subtle fill intensity
    const coresNorm = clamp((selected.vcpu - 1) / (256 - 1), 0, 1);
    const nodeFill = `rgba(80, 130, 220, ${0.08 + coresNorm * 0.25})`;

    const nodeLabelFont = (nodeW: number) => clamp(nodeW * 0.14, 11, 16);

    // DRAM bar geometry inside a node
    const dramH = 18;
    const dramGapAboveLabel = 6;
    const innerPad = 7;
    const dramReserve = dramH + dramGapAboveLabel;

    return (
        <div style={{ display: "grid", gap: 16, padding: 16, maxWidth: 1100, margin: "0 auto" }}>
            <section
                style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 16,
                    alignItems: "start",
                }}
            >
                {/* Left controls */}
                <div
                    style={{
                        border: "1px solid rgba(0,0,0,0.12)",
                        borderRadius: 16,
                        padding: 14,
                        background: "rgba(255,255,255,0.75)",
                    }}
                >
                    <div style={{ display: "grid", gap: 12 }}>
                        <Slider
                            label="Number of nodes"
                            value={config.nodes}
                            min={1}
                            max={256}
                            step={1}
                            testId="nodes-slider"
                            onChange={(nodes) => setConfig((c) => ({ ...c, nodes }))}
                        />

                        <label style={{ display: "grid", gap: 8 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                                <span style={{ fontSize: 13, color: "#111" }}>Instance type</span>
                                <Badge>{selected.id}</Badge>
                            </div>
                            <select
                                value={config.instanceTypeId}
                                onChange={(e) => {
                                    setConfig((c) => ({ ...c, instanceTypeId: e.target.value }));
                                    setIsEditing(false);
                                }}
                                style={{
                                    padding: "10px 10px",
                                    borderRadius: 12,
                                    border: "1px solid rgba(0,0,0,0.14)",
                                    background: "white",
                                    fontSize: 13,
                                }}
                            >
                                {instanceTypes.map((t) => (
                                    <option key={t.id} value={t.id}>
                                        {t.name} (vCPU {t.vcpu}, DRAM {t.dramGb}GB, {t.bandwidthGbps}Gbps)
                                    </option>
                                ))}
                            </select>
                        </label>

                        <div style={{ display: "grid", gap: 10, paddingTop: 6 }}>
                            <button
                                type="button"
                                onClick={() => setIsEditing((v) => !v)}
                                style={{
                                    padding: "10px 12px",
                                    borderRadius: 12,
                                    border: "1px solid rgba(0,0,0,0.14)",
                                    background: "white",
                                    cursor: "pointer",
                                    fontSize: 13,
                                }}
                            >
                                {isEditing ? "Close editor" : "Edit instance"}
                            </button>

                            {isEditing && (
                                <div
                                    style={{
                                        border: "1px solid rgba(0,0,0,0.10)",
                                        borderRadius: 14,
                                        padding: 12,
                                        background: "rgba(255,255,255,0.9)",
                                        display: "grid",
                                        gap: 10,
                                    }}
                                >
                                    <label style={{ display: "grid", gap: 6 }}>
                                        <span style={{ fontSize: 12, color: "#333" }}>Name</span>
                                        <input
                                            value={draft.name}
                                            onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
                                            style={{
                                                padding: "10px 10px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.14)",
                                                fontSize: 13,
                                            }}
                                        />
                                    </label>

                                    <label style={{ display: "grid", gap: 6 }}>
                                        <span style={{ fontSize: 12, color: "#333" }}>vCPU</span>
                                        <input
                                            type="number"
                                            min={1}
                                            max={256}
                                            value={draft.vcpu}
                                            onChange={(e) => setDraft((d) => ({ ...d, vcpu: clamp(Number(e.target.value), 1, 256) }))}
                                            style={{
                                                padding: "10px 10px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.14)",
                                                fontSize: 13,
                                            }}
                                        />
                                    </label>

                                    <label style={{ display: "grid", gap: 6 }}>
                                        <span style={{ fontSize: 12, color: "#333" }}>DRAM (GB)</span>
                                        <input
                                            type="number"
                                            min={1}
                                            max={4096}
                                            value={draft.dramGb}
                                            onChange={(e) => setDraft((d) => ({ ...d, dramGb: clamp(Number(e.target.value), 1, 4096) }))}
                                            style={{
                                                padding: "10px 10px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.14)",
                                                fontSize: 13,
                                            }}
                                        />
                                    </label>

                                    <label style={{ display: "grid", gap: 6 }}>
                                        <span style={{ fontSize: 12, color: "#333" }}>Network bandwidth (Gbps)</span>
                                        <input
                                            type="number"
                                            min={1}
                                            max={400}
                                            value={draft.bandwidthGbps}
                                            onChange={(e) =>
                                                setDraft((d) => ({ ...d, bandwidthGbps: clamp(Number(e.target.value), 1, 400) }))
                                            }
                                            style={{
                                                padding: "10px 10px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.14)",
                                                fontSize: 13,
                                            }}
                                        />
                                    </label>

                                    <div style={{ display: "flex", gap: 10 }}>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setInstanceTypes((types) =>
                                                    types.map((t) =>
                                                        t.id === selected.id
                                                            ? {
                                                                ...t,
                                                                name: draft.name.trim() || t.name,
                                                                vcpu: clamp(draft.vcpu, 1, 256),
                                                                dramGb: clamp(draft.dramGb, 1, 4096),
                                                                bandwidthGbps: clamp(draft.bandwidthGbps, 1, 400),
                                                            }
                                                            : t
                                                    )
                                                );
                                                setIsEditing(false);
                                            }}
                                            style={{
                                                flex: 1,
                                                padding: "10px 12px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.14)",
                                                background: "#fff",
                                                cursor: "pointer",
                                                fontSize: 13,
                                            }}
                                        >
                                            Save instance
                                        </button>

                                        <button
                                            type="button"
                                            onClick={() => {
                                                setDraft({
                                                    name: selected.name,
                                                    vcpu: selected.vcpu,
                                                    dramGb: selected.dramGb,
                                                    bandwidthGbps: selected.bandwidthGbps,
                                                });
                                                setIsEditing(false);
                                            }}
                                            style={{
                                                padding: "10px 12px",
                                                borderRadius: 12,
                                                border: "1px solid rgba(0,0,0,0.10)",
                                                background: "rgba(255,255,255,0.8)",
                                                cursor: "pointer",
                                                fontSize: 13,
                                            }}
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            )}

                            <button
                                type="button"
                                onClick={() => {
                                    setConfig({ nodes: 16, instanceTypeId: "3" });
                                    setInstanceTypes([
                                        { id: "1", name: "Instance 1", vcpu: 4, dramGb: 16, bandwidthGbps: 25 },
                                        { id: "2", name: "Instance 2", vcpu: 8, dramGb: 32, bandwidthGbps: 50 },
                                        { id: "3", name: "Instance 3", vcpu: 16, dramGb: 64, bandwidthGbps: 100 },
                                    ]);
                                    setIsEditing(false);
                                }}
                                style={{
                                    padding: "10px 12px",
                                    borderRadius: 12,
                                    border: "1px solid rgba(0,0,0,0.14)",
                                    background: "white",
                                    cursor: "pointer",
                                    fontSize: 13,
                                }}
                            >
                                Reset
                            </button>
                        </div>
                    </div>
                </div>

                {/* Cluster view (right) */}
                <div
                    className="w-full"
                    style={{
                        border: "1px solid rgba(0,0,0,0.12)",
                        borderRadius: 16,
                        overflow: "hidden",
                        background: "#fafafa",
                    }}
                >
                    <svg width="100%" viewBox="0 0 980 520" role="img" aria-label="Cluster diagram" data-testid="cluster-svg">
                        {/* Links (mesh) */}
                        {viz.links.map((l) => (
                            <g key={l.id}>
                                <line
                                    x1={l.x1}
                                    y1={l.y1}
                                    x2={l.x2}
                                    y2={l.y2}
                                    stroke={l.stroke}
                                    strokeWidth={l.strokeWidth}
                                    opacity={l.opacity}
                                />
                                {l.label && <TextPill x={l.label.x} y={l.label.y} text={l.label.text} />}
                            </g>
                        ))}

                        {/* Nodes */}
                        {viz.nodes.map((n) => {
                            const labelFs = nodeLabelFont(n.w);
                            const dramW = n.w - innerPad * 2;
                            const dramX = n.x + innerPad;
                            const dramY = n.y + n.h - innerPad - 16 - dramH;
                            const labelY = n.y + n.h - innerPad;

                            return (
                                <g key={n.id} data-testid="node">
                                    <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={14} fill="#fff" stroke="#d6d6d6" />
                                    <rect x={n.x + 2} y={n.y + 2} width={n.w - 4} height={n.h - 4} rx={12} fill={nodeFill} opacity={0.85} />

                                    {/* vCPU squares inside the node */}
                                    <VcpuSquares x={n.x} y={n.y} w={n.w} h={n.h} vcpu={selected.vcpu} dramReserve={dramReserve} />

                                    {/* DRAM fixed rectangle */}
                                    <DramBar
                                        x={dramX}
                                        y={dramY}
                                        w={dramW}
                                        h={dramH}
                                        text={`DRAM ${formatNumber(selected.dramGb)} GB`}
                                    />

                                    {/* Node label */}
                                    <text
                                        x={n.x + 8}
                                        y={labelY}
                                        fill="#24344d"
                                        fontSize={labelFs}
                                        style={{ fontVariantNumeric: "tabular-nums" }}
                                    >
                                        N{n.index + 1}
                                    </text>
                                </g>
                            );
                        })}

                        {/* S3 rectangle under all nodes */}
                        <g>
                            <rect
                                x={viz.s3Box.x}
                                y={viz.s3Box.y}
                                width={viz.s3Box.w}
                                height={viz.s3Box.h}
                                rx={16}
                                fill="#ffffff"
                                stroke="#111111"
                                strokeOpacity={0.14}
                            />
                            <text
                                x={viz.s3Box.x + viz.s3Box.w / 2}
                                y={viz.s3Box.y + viz.s3Box.h / 2 + 6}
                                textAnchor="middle"
                                fontSize={18}
                                fill="#222"
                            >
                                S3
                            </text>
                        </g>
                    </svg>
                </div>
            </section>
        </div>
    );
}

