import React, { useMemo } from "react";
import { InstanceConfig } from "@/lib/config";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────

export type ClusterViewProps = {
    nodes: number;
    instanceType: InstanceConfig;
    /** Show the S3 storage box at the bottom (default: true) */
    showS3?: boolean;
    /** Show inter-node mesh links (default: true) */
    showLinks?: boolean;
    /** Custom class name for the wrapper div */
    className?: string;
};

// ─────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────

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
    totals: { totalVcpu: number; totalDramGb: number; totalBandwidthMbps: number };
    warnings: string[];
};

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

function clamp(n: number, min: number, max: number) {
    return Math.max(min, Math.min(max, n));
}

function bandwidthToStrokeWidth(bwGbps: number) {
    const safe = Math.max(1, bwGbps);
    return clamp(Math.log2(safe) * 0.9, 1, 8);
}

function formatNumber(n: number) {
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// ─────────────────────────────────────────────────────────────
// Visualization computation
// ─────────────────────────────────────────────────────────────

function computeClusterViz(
    nodeCount: number,
    instanceType: InstanceConfig,
    svgW: number,
    svgH: number
): ClusterViz {
    const nodesN = clamp(nodeCount, 1, 256);

    const totalVcpu = nodesN * instanceType.cpu_cores;
    const totalDramGb = nodesN * instanceType.memory;
    const totalBandwidthMbps = nodesN * instanceType.network_bandwidth;

    const warnings: string[] = [];
    if (instanceType.network_bandwidth < 5000) warnings.push("Low network bandwidth per node");
    if (nodesN > 160) warnings.push("High node count: inter-node links are thinned for readability/perf");

    // Reserve space for S3 box
    const s3Gap = 18;
    const s3H = 54;

    // Cluster layout area
    const marginX = 50;
    const top = 26;
    const bottom = 20;
    const availW = svgW - marginX * 2;
    const availH = svgH - top - bottom - s3Gap - s3H;

    // Deterministic grid (stable)
    const cols = Math.ceil(Math.sqrt(nodesN));
    const rows = Math.ceil(nodesN / cols);

    const pad = 20;
    const rawW = (availW - (cols - 1) * pad) / cols;
    const rawH = (availH - (rows - 1) * pad) / rows;

    // Node size (big enough to show some vCPU squares + DRAM bar)
    const nodeW = clamp(rawW, 56, 150);
    const nodeH = clamp(rawH, 70, 140);

    const gridW = cols * nodeW + (cols - 1) * pad;
    const gridH = rows * nodeH + (rows - 1) * pad;
    const startX = (svgW - gridW) / 2;
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
        y: Math.min(svgH - bottom - s3H, maxY + s3Gap),
        w: svgW - 140,
        h: s3H,
    };

    // Bandwidth visual encoding on links (from instance type)
    // network_bandwidth is in Mbps, convert to Gbps for visual encoding
    const bwGbps = instanceType.network_bandwidth / 1000;
    const bwStroke = bandwidthToStrokeWidth(bwGbps);
    const bwNorm = clamp(bwGbps / 400, 0, 1);
    const meshStroke = clamp(bwStroke * 0.35, 1, 4);
    const meshOpacity = 0.04 + bwNorm * 0.1;

    // Inter-node links
    const links: LinkViz[] = [];

    const bwLabel = `${(instanceType.network_bandwidth / 1000).toFixed(1)} Gbps`;
    const shouldLabel = (edgeIndex: number, nodesCount: number) => {
        if (nodesCount <= 20) return true;
        if (nodesCount <= 64) return edgeIndex % 2 === 0;
        return edgeIndex % 6 === 0;
    };

    // Full mesh up to 26 nodes
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
            // grid neighbors
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
        totals: { totalVcpu, totalDramGb, totalBandwidthMbps },
        warnings,
    };
}

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function TextPill({ x, y, text }: { x: number; y: number; text: string }) {
    const fontSize = 15;
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
                stroke="#000"
                strokeOpacity={0.08}
            />
            <text x={x} y={y + fontSize * 0.35} textAnchor="middle" fontSize={fontSize} fill="#222">
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
    const margin = 10;
    const gap = 8;

    const labelReserve = 20;
    const availW = Math.max(0, w - margin * 2);
    const availH = Math.max(0, h - margin * 2 - labelReserve - dramReserve);

    const cols = clamp(Math.min(8, vcpu), 1, 8);
    const rowsNeeded = Math.ceil(vcpu / cols);

    const maxBox = 52;
    const minBox = 14;

    const boxByW = (availW - (cols - 1) * gap) / cols;
    const boxByH = (availH - (rowsNeeded - 1) * gap) / Math.max(1, rowsNeeded);
    let box = Math.floor(Math.min(maxBox, boxByW, boxByH));
    box = clamp(box, minBox, maxBox);

    const fitCols = clamp(Math.floor((availW + gap) / (box + gap)), 1, 8);
    const fitRows = clamp(Math.floor((availH + gap) / (box + gap)), 1, 99);
    const maxFit = fitCols * fitRows;

    const left = x + margin;
    const top = y + margin;

    const makeBox = (bx: number, by: number, key: string, text: string) => (
        <g key={key}>
            <rect x={bx} y={by} width={box} height={box} rx={6} fill="#ffffff" stroke="#111111" strokeOpacity={0.22} />
            <text
                x={bx + box / 2}
                y={by + box / 2 + 4}
                textAnchor="middle"
                fontSize={clamp(box * 0.42, 10, 18)}
                fill="#222"
            >
                {text}
            </text>
        </g>
    );

    if (maxFit <= 1) {
        const txt = vcpu >= 1000 ? `+${Math.floor(vcpu / 1000)}k` : `+${vcpu}`;
        return <g>{makeBox(left, top, "v-sum", vcpu === 1 ? "vCPU" : txt)}</g>;
    }

    const renderCount = vcpu <= maxFit ? vcpu : Math.max(0, maxFit - 1);
    const remaining = vcpu - renderCount;

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
            <text x={x + w / 2} y={y + h / 2 + 6} textAnchor="middle" fontSize={17} fill="#222" opacity={0.9}>
                {text}
            </text>
        </g>
    );
}

// ─────────────────────────────────────────────────────────────
// Main ClusterView component
// ─────────────────────────────────────────────────────────────

export function ClusterView({
    nodes,
    instanceType,
    showS3 = true,
    showLinks = true,
    className,
}: ClusterViewProps) {
    // Fixed viewBox dimensions for consistent aspect ratio
    const viewBoxWidth = 980;
    const viewBoxHeight = 820;

    const viz = useMemo(
        () => computeClusterViz(nodes, instanceType, viewBoxWidth, viewBoxHeight),
        [nodes, instanceType]
    );

    // vCPU -> subtle fill intensity
    const coresNorm = clamp((instanceType.cpu_cores - 1) / (256 - 1), 0, 1);
    const nodeFill = `rgba(80, 130, 220, ${0.08 + coresNorm * 0.25})`;

    const nodeLabelFont = (nodeW: number) => clamp(nodeW * 0.18, 15, 22);

    // DRAM bar geometry inside a node
    const dramH = 28;
    const dramGapAboveLabel = 10;
    const innerPad = 10;
    const dramReserve = dramH + dramGapAboveLabel;

    return (
        <div
            className={`w-full ${className || ""}`}
            style={{
                border: "1px solid rgba(0,0,0,0.12)",
                borderRadius: 16,
                overflow: "hidden",
            }}
        >
            <svg
                className="w-full h-full"
                viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
                preserveAspectRatio="xMidYMid meet"
                role="img"
                aria-label="Cluster diagram"
                data-testid="cluster-svg"
            >
                {/* Links (mesh) */}
                {showLinks &&
                    viz.links.map((l) => (
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
                            <rect
                                x={n.x + 2}
                                y={n.y + 2}
                                width={n.w - 4}
                                height={n.h - 4}
                                rx={12}
                                fill={nodeFill}
                            />

                            {/* vCPU squares inside the node */}
                            <VcpuSquares
                                x={n.x}
                                y={n.y}
                                w={n.w}
                                h={n.h}
                                vcpu={instanceType.cpu_cores}
                                dramReserve={dramReserve}
                            />

                            {/* DRAM bar */}
                            <DramBar
                                x={dramX}
                                y={dramY}
                                w={dramW}
                                h={dramH}
                                text={`DRAM ${formatNumber(instanceType.memory)} GB`}
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
                {showS3 && (
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
                )}
            </svg>
        </div>
    );
}

export default ClusterView;
