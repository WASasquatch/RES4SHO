// SigmaCurves -- step-locked sigma editor.
//
// One plot point per sampling step (steps + 1 points total). X positions
// are fixed; only y is draggable. The chosen scheduler seeds the plot
// shape (fetched from /RES4SHO/sigma_curves/preview). Range selection +
// "apply curve to range" overwrites the y values across that range using
// the selected interpolation, so a single schedule can mix multiple
// curve archetypes (sigmoid head, bezier middle, step tail, etc.).

import { app } from "../../scripts/app.js";

// ----- Interpolators (used only for "apply curve to range") ----------

// "custom" is the no-op option: picking any other curve immediately
// reshapes the selected range (or the whole curve, if no range is set)
// using that interpolation. Picking "custom" leaves edits alone.
const INTERP_OPTIONS = [
    "custom",
    "linear", "step", "step_next", "smoothstep", "smootherstep", "cosine",
    "sigmoid", "atan", "ease_in", "ease_out", "ease_in_out", "exp",
];

const DEFAULT_K = {
    sigmoid: 8, atan: 6,
    ease_in: 2, ease_out: 2, ease_in_out: 3, exp: 4,
};

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function segLerp(y0, y1, u)         { return y0 + (y1 - y0) * u; }
function segStep(y0, y1, u)         { return y0; }
function segStepNext(y0, y1, u)     { return y1; }
function segSmoothstep(y0, y1, u)   { const s = u*u*(3-2*u); return y0 + (y1-y0)*s; }
function segSmootherstep(y0, y1, u) { const s = u*u*u*(u*(u*6-15)+10); return y0 + (y1-y0)*s; }
function segCosine(y0, y1, u)       { const s = (1 - Math.cos(Math.PI*u)) * 0.5; return y0 + (y1-y0)*s; }
function segSigmoid(y0, y1, u, k) {
    if (k <= 1e-6) return y0 + (y1-y0)*u;
    const sRaw = 1/(1+Math.exp(-k*(u-0.5)));
    const sMin = 1/(1+Math.exp(k*0.5));
    const sMax = 1/(1+Math.exp(-k*0.5));
    const s = (sRaw - sMin) / Math.max(sMax - sMin, 1e-12);
    return y0 + (y1-y0)*s;
}
function segAtan(y0, y1, u, k) {
    if (k <= 1e-6) return y0 + (y1-y0)*u;
    const denom = Math.atan(k*0.5);
    if (denom < 1e-12) return y0 + (y1-y0)*u;
    const s = (Math.atan(k*(u-0.5))/denom + 1) * 0.5;
    return y0 + (y1-y0)*s;
}
function segEaseIn(y0, y1, u, k)    { return y0 + (y1-y0) * Math.pow(u, Math.max(k, 0.01)); }
function segEaseOut(y0, y1, u, k)   { return y0 + (y1-y0) * (1 - Math.pow(1-u, Math.max(k, 0.01))); }
function segEaseInOut(y0, y1, u, k) {
    const kk = Math.max(k, 0.01);
    const s = u < 0.5 ? 0.5*Math.pow(2*u, kk) : 1 - 0.5*Math.pow(2*(1-u), kk);
    return y0 + (y1-y0)*s;
}
function segExp(y0, y1, u, k) {
    if (Math.abs(k) < 1e-6) return y0 + (y1-y0)*u;
    const s = (Math.exp(k*u) - 1) / (Math.exp(k) - 1);
    return y0 + (y1-y0)*s;
}

const SEG_FNS = {
    linear: segLerp, step: segStep, step_next: segStepNext,
    smoothstep: segSmoothstep, smootherstep: segSmootherstep,
    cosine: segCosine, sigmoid: segSigmoid, atan: segAtan,
    ease_in: segEaseIn, ease_out: segEaseOut, ease_in_out: segEaseInOut,
    exp: segExp,
};

function applyRangeCurve(values, a, b, interp, tension) {
    if (!interp || interp === "custom") return;
    if (a === b) return;
    if (a > b) { const t = a; a = b; b = t; }
    const yA = values[a], yB = values[b];
    const span = b - a;
    const k = (tension && tension !== 0) ? tension : (DEFAULT_K[interp] || 0);
    const fn = SEG_FNS[interp] || segLerp;
    for (let i = a + 1; i < b; i++) {
        const u = (i - a) / span;
        values[i] = clamp(fn(yA, yB, u, k), 0, 1);
    }
}

// ----- Server fetch ---------------------------------------------------

async function fetchBaseline(scheduler, steps) {
    try {
        const url = `/RES4SHO/sigma_curves/preview?scheduler=${
            encodeURIComponent(scheduler)}&steps=${steps}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error("HTTP " + r.status);
        const data = await r.json();
        // Surface dispatch + any fallback/error info to the console so
        // the user can verify their schedulers are actually being run
        // (vs. silently falling back to a linear stub).
        const tag = `[SigmaCurves] '${scheduler}' x${steps}`;
        if (data.fallback || data.error) {
            console.warn(`${tag}  dispatch=${data.dispatch || "?"}  `,
                         data.error ? `error=${data.error}` : "fallback");
        } else {
            console.debug(`${tag}  dispatch=${data.dispatch || "?"}`);
        }
        if (Array.isArray(data.values) && data.values.length >= 2) {
            let arr = data.values.map(v => clamp(+v, 0, 1));
            // Some schedulers return steps+2 (or other lengths). Resample
            // to exactly steps+1 here as a safety net; the backend now
            // does this too, but we keep a client-side fallback so any
            // future scheduler quirk doesn't silently fall through to
            // the linear default.
            const target = steps + 1;
            if (arr.length !== target) {
                const old = arr;
                const out = [];
                for (let i = 0; i < target; i++) {
                    const t = i / Math.max(target - 1, 1) * (old.length - 1);
                    const lo = Math.floor(t);
                    const hi = Math.min(lo + 1, old.length - 1);
                    const frac = t - lo;
                    out.push(old[lo] * (1 - frac) + old[hi] * frac);
                }
                if (data.trailing_zero) out[out.length - 1] = 0;
                arr = out;
                console.warn(
                    `[SigmaCurves] resampled ${old.length} -> ${target} values`);
            }
            return {
                values: arr,
                trailing_zero: !!data.trailing_zero,
                fallback: !!data.fallback,
                dispatch: data.dispatch || "unknown",
            };
        }
        throw new Error("invalid response shape");
    } catch (e) {
        console.warn("SigmaCurves: baseline fetch failed, using linear", e);
        const v = [];
        for (let i = 0; i <= steps; i++) v.push(1 - i / Math.max(steps, 1));
        return { values: v, trailing_zero: true, fallback: true,
                 dispatch: "frontend_linear_fallback" };
    }
}

// ----- Widget --------------------------------------------------------

const HEIGHT = 280;
const PAD_L = 42, PAD_R = 12, PAD_B = 26;
const POINT_R = 4;
const HIT_R = 8;

// Toolbar: header text at y=2, then a strip at y=18 with curve/tension/apply
// buttons. Plot starts at y=44 (PLOT_TOP) leaving the toolbar room above.
const HEADER_Y = 2;
const TOOLBAR_Y = 18;
const TOOLBAR_H = 22;
const PLOT_TOP = TOOLBAR_Y + TOOLBAR_H + 4;

function toolbarRects(widgetWidth) {
    const y = TOOLBAR_Y, h = TOOLBAR_H, x0 = PAD_L;
    return {
        y, h,
        curve:   { x: x0,        y, w: 130, h },
        tension: { x: x0 + 136,  y, w: 60,  h },
        apply:   { x: x0 + 200,  y, w: 70,  h },
    };
}

function inRect(r, x, y) {
    return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
}

function drawToolbarButton(ctx, r, ox, oy, label, active, hover) {
    ctx.fillStyle = hover ? "#333" : "#222";
    ctx.fillRect(ox + r.x, oy + r.y, r.w, r.h);
    ctx.strokeStyle = active ? "#5cf" : "#3a3a3a";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + r.x + 0.5, oy + r.y + 0.5, r.w - 1, r.h - 1);
    ctx.fillStyle = active ? "#fff" : "#aaa";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, ox + r.x + r.w / 2, oy + r.y + r.h / 2);
}

function makeStepCurveWidget(node, schedulerWidget, stepsWidget, dataWidget) {
    const state = {
        values: null,        // y values, one per step+1; null until fetched
        steps: stepsWidget?.value || 20,
        scheduler: schedulerWidget?.value || "normal",
        dragging: -1,
        rightDragging: false,
        hover: -1,
        toolbarHover: null,  // "curve" | "tension" | "apply" | null
        selStart: -1,
        selEnd: -1,
        interp: "custom",
        tension: 0,
        lastValueSeen: null,
        lastFetched: null,   // {scheduler, steps} of the last successful fetch
    };

    function syncFromDataWidget() {
        const v = dataWidget.value;
        if (v === state.lastValueSeen) return;
        state.lastValueSeen = v;
        if (!v) return;
        try {
            const obj = JSON.parse(v);
            if (Array.isArray(obj.values) && obj.values.length >= 2) {
                state.values = obj.values.map(x => clamp(+x, 0, 1));
            }
            if (typeof obj.scheduler === "string") state.scheduler = obj.scheduler;
            if (typeof obj.steps === "number") state.steps = obj.steps | 0;
            if (typeof obj.interp === "string") state.interp = obj.interp;
            if (typeof obj.tension === "number") state.tension = obj.tension;
        } catch (e) { /* keep what we have */ }
    }

    function pushToDataWidget() {
        if (!state.values) return;
        const obj = {
            values: state.values.map(v => +(+v).toFixed(6)),
            scheduler: state.scheduler,
            steps: state.steps,
            interp: state.interp,
            tension: state.tension,
        };
        const json = JSON.stringify(obj);
        dataWidget.value = json;
        state.lastValueSeen = json;
        node.setDirtyCanvas(true, true);
    }

    // Apply the current toolbar interp/tension to the active range.
    // The "active range" is the current selection if one exists, else
    // the whole curve. Endpoints are preserved so the curve fits through
    // the existing y[start] -> y[end].
    function applyToSelection() {
        if (!state.values) return;
        if (!state.interp || state.interp === "custom") return;
        let lo, hi;
        if (state.selStart >= 0 && state.selEnd >= 0) {
            lo = Math.min(state.selStart, state.selEnd);
            hi = Math.max(state.selStart, state.selEnd);
        } else {
            lo = 0;
            hi = state.values.length - 1;
        }
        if (hi - lo < 2) return;
        applyRangeCurve(state.values, lo, hi, state.interp, state.tension);
        pushToDataWidget();
    }

    function showCurveDropdown(event) {
        const choices = INTERP_OPTIONS.slice();
        if (typeof LiteGraph !== "undefined" && LiteGraph.ContextMenu) {
            new LiteGraph.ContextMenu(choices, {
                event,
                callback: (selected) => {
                    if (typeof selected === "string") {
                        state.interp = selected;
                        pushToDataWidget();
                    }
                },
            });
        } else {
            // Fallback if LiteGraph.ContextMenu isn't available.
            const v = window.prompt(
                `Curve type (one of: ${choices.join(", ")}):`, state.interp);
            if (v && choices.includes(v)) {
                state.interp = v;
                pushToDataWidget();
            }
        }
    }

    function promptTension() {
        const v = window.prompt("Tension (0-30):", String(state.tension));
        if (v === null) return;
        const num = parseFloat(v);
        if (!isNaN(num)) {
            state.tension = clamp(num, 0, 30);
            pushToDataWidget();
        }
    }

    async function refreshBaseline() {
        const sch = schedulerWidget?.value || "normal";
        const stp = stepsWidget?.value || 20;
        const result = await fetchBaseline(sch, stp);
        state.scheduler = sch;
        state.steps = stp;
        state.values = result.values;
        state.selStart = state.selEnd = -1;
        state.lastFetched = { scheduler: sch, steps: stp };
        pushToDataWidget();
    }

    function resampleToSteps() {
        const stp = stepsWidget?.value || 20;
        const target = stp + 1;
        if (!state.values) {
            state.steps = stp;
            return;
        }
        if (state.values.length === target) {
            state.steps = stp;
            return;
        }
        const old = state.values;
        const nNew = target;
        const nOld = old.length;
        const out = [];
        for (let i = 0; i < nNew; i++) {
            const t = i / Math.max(nNew - 1, 1) * (nOld - 1);
            const lo = Math.floor(t);
            const hi = Math.min(lo + 1, nOld - 1);
            const frac = t - lo;
            out.push(old[lo] * (1 - frac) + old[hi] * frac);
        }
        state.values = out;
        state.steps = stp;
        state.selStart = state.selEnd = -1;
        pushToDataWidget();
    }

    function watchWidgets() {
        if (schedulerWidget) {
            const orig = schedulerWidget.callback;
            schedulerWidget.callback = function(v, ...rest) {
                const r = orig?.apply(this, [v, ...rest]);
                refreshBaseline();
                return r;
            };
        }
        if (stepsWidget) {
            const orig = stepsWidget.callback;
            stepsWidget.callback = function(v, ...rest) {
                const r = orig?.apply(this, [v, ...rest]);
                // If the steps changed and the user hasn't edited from
                // the last-fetched baseline, refetch (so the shape stays
                // accurate to the scheduler at the new step count).
                // Otherwise, resample existing edits.
                if (state.lastFetched
                    && state.lastFetched.scheduler === schedulerWidget?.value) {
                    refreshBaseline();
                } else {
                    resampleToSteps();
                }
                return r;
            };
        }
    }

    function plotRect(widgetWidth) {
        return {
            x: PAD_L,
            y: PLOT_TOP,
            w: Math.max(20, widgetWidth - PAD_L - PAD_R),
            h: Math.max(20, HEIGHT - PLOT_TOP - PAD_B),
        };
    }
    function dataToPlot(rect, t, y) {
        return [rect.x + t * rect.w, rect.y + (1 - y) * rect.h];
    }
    function plotToValue(rect, py) {
        return clamp(1 - (py - rect.y) / rect.h, 0, 1);
    }

    function findToolbarHit(widgetWidth, localX, localY) {
        const tb = toolbarRects(widgetWidth);
        if (localY < tb.y || localY > tb.y + tb.h) return null;
        if (inRect(tb.curve, localX, localY)) return "curve";
        if (inRect(tb.tension, localX, localY)) return "tension";
        if (inRect(tb.apply, localX, localY)) return "apply";
        return null;
    }

    function findStep(rect, px, py) {
        if (!state.values) return -1;
        const n = state.values.length;
        const t = clamp((px - rect.x) / rect.w, 0, 1);
        const stepF = t * (n - 1);
        const stepIdx = Math.round(stepF);
        const [hx, hy] = dataToPlot(rect, stepIdx / (n - 1), state.values[stepIdx]);
        const dx = px - hx, dy = py - hy;
        return (dx*dx + dy*dy <= HIT_R*HIT_R) ? stepIdx : -1;
    }

    function inSelectedRange(idx) {
        if (state.selStart < 0 || state.selEnd < 0) return false;
        const lo = Math.min(state.selStart, state.selEnd);
        const hi = Math.max(state.selStart, state.selEnd);
        return idx >= lo && idx <= hi;
    }

    const widget = {
        type: "sigma_curve_steps",
        name: "sigma_curve_canvas",
        options: { serialize: false },
        last_y: 0,

        draw(ctx, gnode, widgetWidth, y, widgetHeight) {
            this.last_y = y;
            syncFromDataWidget();

            const rect = plotRect(widgetWidth);
            const ox = 0, oy = y;

            ctx.save();
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(ox + rect.x, oy + rect.y, rect.w, rect.h);

            // Grid
            ctx.strokeStyle = "#2e2e2e";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i <= 10; i++) {
                const gx = rect.x + (i / 10) * rect.w;
                ctx.moveTo(ox + gx, oy + rect.y);
                ctx.lineTo(ox + gx, oy + rect.y + rect.h);
            }
            for (let i = 0; i <= 5; i++) {
                const gy = rect.y + (i / 5) * rect.h;
                ctx.moveTo(ox + rect.x, oy + gy);
                ctx.lineTo(ox + rect.x + rect.w, oy + gy);
            }
            ctx.stroke();

            if (!state.values) {
                ctx.fillStyle = "#888";
                ctx.font = "12px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("loading scheduler baseline…",
                             ox + rect.x + rect.w * 0.5,
                             oy + rect.y + rect.h * 0.5);
                ctx.restore();
                return;
            }

            const n = state.values.length;

            // Selected range fill
            if (state.selStart >= 0 && state.selEnd >= 0) {
                const lo = Math.min(state.selStart, state.selEnd);
                const hi = Math.max(state.selStart, state.selEnd);
                const xL = rect.x + (lo / (n - 1)) * rect.w;
                const xR = rect.x + (hi / (n - 1)) * rect.w;
                ctx.fillStyle = "rgba(255, 220, 0, 0.10)";
                ctx.fillRect(ox + xL, oy + rect.y, Math.max(xR - xL, 1), rect.h);
            }

            // Axis labels
            ctx.fillStyle = "#888";
            ctx.font = "10px monospace";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText("σ_max", ox + rect.x - 4, oy + rect.y);
            ctx.fillText("0.5",   ox + rect.x - 4, oy + rect.y + rect.h * 0.5);
            ctx.fillText("σ_min", ox + rect.x - 4, oy + rect.y + rect.h);
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillText("step 0", ox + rect.x, oy + rect.y + rect.h + 4);
            ctx.fillText(`step ${n - 1}`, ox + rect.x + rect.w,
                         oy + rect.y + rect.h + 4);

            // Curve polyline through values
            ctx.strokeStyle = "#5cf";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const t = i / (n - 1);
                const [px, py] = dataToPlot(rect, t, state.values[i]);
                if (i === 0) ctx.moveTo(ox + px, oy + py);
                else ctx.lineTo(ox + px, oy + py);
            }
            ctx.stroke();

            // Per-step dots
            for (let i = 0; i < n; i++) {
                const t = i / (n - 1);
                const [px, py] = dataToPlot(rect, t, state.values[i]);
                const isInRange = inSelectedRange(i);
                const isHover = (state.hover === i || state.dragging === i);
                const isAnchor = isInRange && (i === Math.min(state.selStart, state.selEnd)
                                               || i === Math.max(state.selStart, state.selEnd));
                ctx.beginPath();
                ctx.arc(ox + px, oy + py,
                        isHover ? POINT_R + 1.5 : POINT_R,
                        0, Math.PI * 2);
                if (isAnchor) ctx.fillStyle = "#fc0";
                else if (isInRange) ctx.fillStyle = "#ff8";
                else ctx.fillStyle = "#5cf";
                ctx.fill();
                ctx.strokeStyle = "#000";
                ctx.lineWidth = 1;
                ctx.stroke();
            }

            // Hover label (after dots so it sits on top)
            if (state.hover >= 0 && state.values[state.hover] !== undefined) {
                const i = state.hover;
                const t = i / (n - 1);
                const [px, py] = dataToPlot(rect, t, state.values[i]);
                ctx.font = "10px monospace";
                const txt = `step ${i}: ${state.values[i].toFixed(3)}`;
                const w = ctx.measureText(txt).width + 8;
                let lx = ox + px - w * 0.5;
                if (lx < ox + rect.x) lx = ox + rect.x;
                if (lx + w > ox + rect.x + rect.w) lx = ox + rect.x + rect.w - w;
                ctx.fillStyle = "rgba(0,0,0,0.85)";
                ctx.fillRect(lx, oy + py - 22, w, 14);
                ctx.fillStyle = "#fff";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(txt, lx + w * 0.5, oy + py - 15);
            }

            // Header text (top of widget, above the toolbar)
            ctx.font = "10px monospace";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            const sel = (state.selStart >= 0 && state.selEnd >= 0)
                ? `range [${Math.min(state.selStart, state.selEnd)}..${Math.max(state.selStart, state.selEnd)}]`
                : "no range";
            ctx.fillStyle = (state.selStart >= 0) ? "#fc0" : "#bbb";
            ctx.fillText(`${state.scheduler} | ${n - 1} steps | ${sel}`,
                         ox + rect.x, oy + HEADER_Y);
            ctx.textAlign = "right";
            ctx.fillStyle = "#777";
            ctx.fillText("L-drag y=adjust   R-drag=select range",
                         ox + rect.x + rect.w, oy + HEADER_Y);

            // Toolbar -- the in-canvas curve picker, tension input, apply.
            const tb = toolbarRects(widgetWidth);
            const canApply = state.interp !== "custom"
                && state.values
                && (() => {
                    const lo = (state.selStart >= 0 && state.selEnd >= 0)
                        ? Math.min(state.selStart, state.selEnd) : 0;
                    const hi = (state.selStart >= 0 && state.selEnd >= 0)
                        ? Math.max(state.selStart, state.selEnd) : state.values.length - 1;
                    return hi - lo >= 2;
                })();

            drawToolbarButton(ctx, tb.curve, ox, oy,
                              `${state.interp} ▾`,
                              state.interp !== "custom",
                              state.toolbarHover === "curve");
            drawToolbarButton(ctx, tb.tension, ox, oy,
                              `k ${state.tension.toFixed(2)}`,
                              state.tension !== 0,
                              state.toolbarHover === "tension");
            drawToolbarButton(ctx, tb.apply, ox, oy,
                              "apply",
                              canApply,
                              canApply && state.toolbarHover === "apply");

            ctx.restore();
        },

        mouse(event, pos, gnode) {
            if (!state.values) return false;
            const rect = plotRect(gnode.size[0]);
            const localX = pos[0];
            const localY = pos[1] - this.last_y;
            // Extend the hit area horizontally by HIT_R + a couple of pixels
            // so clicks on the LEFT half of step 0's dot (centered at
            // rect.x) and the RIGHT half of step N's dot (centered at
            // rect.x + rect.w) still register. Without this, the leftmost
            // and rightmost steps cannot be selected by shift-click /
            // dragged because their dots straddle the plot rect edge.
            const HM = HIT_R + 2;
            const inPlot = localX >= rect.x - HM && localX <= rect.x + rect.w + HM &&
                           localY >= rect.y && localY <= rect.y + rect.h;

            const evType = event.type;
            const button = (event.button !== undefined) ? event.button : 0;

            // Translate cursor X to the nearest step index.
            const stepFromX = (px) => {
                const n = state.values.length;
                const t = clamp((px - rect.x) / rect.w, 0, 1);
                return Math.round(t * (n - 1));
            };

            if (evType === "pointerdown" || evType === "mousedown") {
                // In-canvas toolbar takes priority over plot interactions.
                if (button === 0) {
                    const tbHit = findToolbarHit(gnode.size[0], localX, localY);
                    if (tbHit === "curve") {
                        showCurveDropdown(event);
                        return true;
                    }
                    if (tbHit === "tension") {
                        promptTension();
                        return true;
                    }
                    if (tbHit === "apply") {
                        applyToSelection();
                        return true;
                    }
                }

                if (!inPlot) return false;

                // Right-button anywhere in the plot: start a range drag.
                // Shift+left also extends a range, kept as a fallback.
                if (button === 2) {
                    const idx = stepFromX(localX);
                    state.selStart = idx;
                    state.selEnd = idx;
                    state.rightDragging = true;
                    if (typeof window !== "undefined") {
                        window.__res4sho_suppress_ctxmenu = true;
                    }
                    node.setDirtyCanvas(true, true);
                    event.preventDefault?.();
                    event.stopPropagation?.();
                    return true;
                }

                if (event.shiftKey) {
                    const idx = stepFromX(localX);
                    if (state.selStart < 0) state.selStart = idx;
                    state.selEnd = idx;
                    node.setDirtyCanvas(true, true);
                    return true;
                }

                // Plain left-click: drag the y of the nearest dot if it
                // was clicked on; otherwise no-op (leaves the selection
                // intact so the user can apply curves repeatedly).
                const idx = findStep(rect, localX, localY);
                if (idx >= 0) {
                    state.dragging = idx;
                    state.hover = idx;
                    return true;
                }
                return inPlot;
            }

            if (evType === "pointermove" || evType === "mousemove") {
                // Update toolbar hover (works above the plot rect too).
                const tbHover = findToolbarHit(gnode.size[0], localX, localY);
                if (tbHover !== state.toolbarHover) {
                    state.toolbarHover = tbHover;
                    node.setDirtyCanvas(true, true);
                }
                // Right-drag to extend the range.
                if (state.rightDragging) {
                    state.selEnd = stepFromX(localX);
                    node.setDirtyCanvas(true, true);
                    event.preventDefault?.();
                    return true;
                }
                // Left-drag to update the y of the held dot.
                if (state.dragging >= 0) {
                    state.values[state.dragging] = plotToValue(rect, localY);
                    pushToDataWidget();
                    return true;
                }
                const newHover = inPlot ? findStep(rect, localX, localY) : -1;
                if (newHover !== state.hover) {
                    state.hover = newHover;
                    node.setDirtyCanvas(true, true);
                }
                return tbHover !== null || inPlot;
            }

            if (evType === "pointerup" || evType === "mouseup") {
                if (state.rightDragging) {
                    state.rightDragging = false;
                    event.preventDefault?.();
                    return true;
                }
                if (state.dragging >= 0) {
                    state.dragging = -1;
                    pushToDataWidget();
                    return true;
                }
                return false;
            }

            // Suppress LiteGraph / browser context menus inside the plot
            // so the right-button drag works cleanly.
            if (evType === "contextmenu") {
                if (inPlot) {
                    event.preventDefault?.();
                    event.stopPropagation?.();
                    return true;
                }
                return false;
            }

            return false;
        },

        computeSize(width) { return [Math.max(width, 320), HEIGHT]; },
        serializeValue() { return null; },

        // ---- Right-click drag entry points ----
        // LiteGraph's canvas short-circuits right-click directly into the
        // context-menu path and does NOT forward those events to widget
        // mouse callbacks. The document-level listener installed below
        // calls these methods instead, with widget-local coordinates
        // already resolved.
        _sigmaRightDown(localX, localY, gnode) {
            if (!state.values) return false;
            const rect = plotRect(gnode.size[0]);
            const HM = HIT_R + 2;
            if (localX < rect.x - HM || localX > rect.x + rect.w + HM ||
                localY < rect.y || localY > rect.y + rect.h) return false;
            const n = state.values.length;
            const t = clamp((localX - rect.x) / rect.w, 0, 1);
            const idx = Math.round(t * (n - 1));
            state.selStart = idx;
            state.selEnd = idx;
            state.rightDragging = true;
            node.setDirtyCanvas(true, true);
            return true;
        },
        _sigmaRightMove(localX, localY, gnode) {
            if (!state.rightDragging || !state.values) return false;
            const rect = plotRect(gnode.size[0]);
            const n = state.values.length;
            const t = clamp((localX - rect.x) / rect.w, 0, 1);
            state.selEnd = Math.round(t * (n - 1));
            node.setDirtyCanvas(true, true);
            return true;
        },
        _sigmaRightUp() {
            if (!state.rightDragging) return false;
            state.rightDragging = false;
            return true;
        },
    };

    // Curve / tension / apply UI lives inside the plot widget itself --
    // the in-canvas toolbar drawn above the plot. See draw() and mouse()
    // for the rendering and hit-testing of those controls.
    node.addWidget(
        "button", "select all steps", null,
        () => {
            if (!state.values) return;
            state.selStart = 0;
            state.selEnd = state.values.length - 1;
            node.setDirtyCanvas(true, true);
        },
        { serialize: false },
    );
    node.addWidget(
        "button", "clear range selection", null,
        () => {
            state.selStart = state.selEnd = -1;
            node.setDirtyCanvas(true, true);
        },
        { serialize: false },
    );
    node.addWidget(
        "button", "reset to scheduler default", null,
        () => refreshBaseline(),
        { serialize: false },
    );
    node.addWidget(
        "button", "flatten range to start value", null,
        () => {
            if (!state.values || state.selStart < 0 || state.selEnd < 0) return;
            const lo = Math.min(state.selStart, state.selEnd);
            const hi = Math.max(state.selStart, state.selEnd);
            const v = state.values[lo];
            for (let i = lo; i <= hi; i++) state.values[i] = v;
            pushToDataWidget();
        },
        { serialize: false },
    );

    // Initial population: prefer saved curve_data, else fetch fresh.
    syncFromDataWidget();
    if (!state.values || state.values.length !== (stepsWidget?.value || 20) + 1) {
        refreshBaseline();
    } else {
        pushToDataWidget();
    }

    watchWidgets();
    return widget;
}

// One-shot install: capture-phase pointer / contextmenu listeners that
// bypass LiteGraph's right-click handling so we can implement
// right-button drag for range selection on SigmaCurves nodes.
//
// LiteGraph's `LGraphCanvas.processMouseDown` short-circuits right-click
// straight into context-menu logic and never forwards those events to
// `widget.mouse`, which is why the previous in-widget right-click handler
// did nothing in practice. By listening on `document` in capture phase
// we get the events first, find the SigmaCurves node + plot widget under
// the cursor, route the event to widget methods that maintain the range
// selection state, and call `preventDefault` + `stopPropagation` so
// LiteGraph never sees them.
if (typeof window !== "undefined" && !window.__res4sho_events_installed) {
    window.__res4sho_events_installed = true;

    function _findSigmaPlot(e) {
        const cv = app.canvas?.canvas;
        if (!cv) return null;
        const r = cv.getBoundingClientRect();
        const cx = e.clientX - r.left;
        const cy = e.clientY - r.top;
        const ds = app.canvas.ds;
        if (!ds) return null;
        const gx = (cx - ds.offset[0]) / ds.scale;
        const gy = (cy - ds.offset[1]) / ds.scale;
        const nodes = app.graph?._nodes || [];
        for (const n of nodes) {
            if (n.type !== "SigmaCurves") continue;
            // Title bar typically extends ~30px above pos. Match LiteGraph's
            // own bounding-box test loosely.
            const titleH = 30;
            if (gx < n.pos[0] || gx > n.pos[0] + n.size[0]) continue;
            if (gy < n.pos[1] - titleH || gy > n.pos[1] + n.size[1]) continue;
            const w = (n.widgets || []).find(
                (w) => w && w.type === "sigma_curve_steps");
            if (!w || w.last_y == null) continue;
            const localX = gx - n.pos[0];
            const localY = gy - n.pos[1] - w.last_y;
            return { node: n, widget: w, localX, localY };
        }
        return null;
    }

    let _activeRight = null;

    document.addEventListener("pointerdown", (e) => {
        if (e.button !== 2) return;
        const hit = _findSigmaPlot(e);
        if (!hit) return;
        const handled = hit.widget._sigmaRightDown?.(
            hit.localX, hit.localY, hit.node);
        if (handled) {
            _activeRight = hit;
            window.__res4sho_suppress_ctxmenu = true;
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);

    document.addEventListener("pointermove", (e) => {
        if (!_activeRight) return;
        // Re-resolve coords so the user can drag across the plot even if
        // the cursor briefly leaves and re-enters; clamp to the original
        // node's rect via the widget's own clamping in _sigmaRightMove.
        const cv = app.canvas?.canvas;
        if (!cv) return;
        const r = cv.getBoundingClientRect();
        const ds = app.canvas.ds;
        if (!ds) return;
        const gx = ((e.clientX - r.left) - ds.offset[0]) / ds.scale;
        const gy = ((e.clientY - r.top) - ds.offset[1]) / ds.scale;
        const localX = gx - _activeRight.node.pos[0];
        const localY = gy - _activeRight.node.pos[1] - _activeRight.widget.last_y;
        _activeRight.widget._sigmaRightMove?.(localX, localY, _activeRight.node);
        e.preventDefault();
    }, true);

    document.addEventListener("pointerup", (e) => {
        if (!_activeRight) return;
        _activeRight.widget._sigmaRightUp?.();
        _activeRight = null;
        e.preventDefault();
        e.stopPropagation();
    }, true);

    // Swallow the contextmenu that follows a right-click drag.
    document.addEventListener("contextmenu", (e) => {
        if (window.__res4sho_suppress_ctxmenu) {
            e.preventDefault();
            e.stopPropagation();
            window.__res4sho_suppress_ctxmenu = false;
        }
    }, true);
}

app.registerExtension({
    name: "RES4SHO.SigmaCurves",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "SigmaCurves") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const dataWidget = this.widgets?.find(w => w.name === "curve_data");
            const schedulerWidget = this.widgets?.find(w => w.name === "scheduler");
            const stepsWidget = this.widgets?.find(w => w.name === "steps");
            if (!dataWidget) return r;

            dataWidget.type = "hidden";
            dataWidget.computeSize = () => [0, -4];

            const w = makeStepCurveWidget(this, schedulerWidget, stepsWidget, dataWidget);
            this.addCustomWidget(w);

            const natural = this.computeSize?.() || [380, 540];
            this.size = [Math.max(natural[0], 380), natural[1]];
            this.setDirtyCanvas?.(true, true);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            this.setDirtyCanvas?.(true, true);
            return r;
        };
    },
});
