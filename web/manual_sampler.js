// ManualSampler -- save / load / delete preset UI for the Manual
// Sampler node. The node itself is a regular widget-only node; the
// only frontend concern is the preset management (mirrors the Sigma
// Curves preset UI).

import { app } from "../../scripts/app.js";

// ----- Themed dialogs (same primitives as sigma_curves.js) -----------

const _DIALOG_BG = "var(--p-overlay-modal-background, var(--comfy-menu-bg, #2a2a2a))";
const _DIALOG_FG = "var(--p-text-color, var(--fg-color, #eee))";
const _DIALOG_BORDER = "var(--p-overlay-modal-border-color, var(--border-color, #444))";
const _INPUT_BG = "var(--p-form-field-background, var(--comfy-input-bg, #1a1a1a))";
const _INPUT_FG = "var(--p-form-field-color, var(--input-text, #eee))";
const _INPUT_BORDER = "var(--p-form-field-border-color, var(--border-color, #555))";
const _BTN_BG = "var(--p-button-secondary-background, var(--comfy-input-bg, #3a3a3a))";
const _BTN_FG = "var(--p-button-secondary-color, var(--fg-color, #eee))";
const _BTN_BORDER = "var(--p-button-secondary-border-color, var(--border-color, #555))";
const _BTN_PRIMARY_BG = "var(--p-button-primary-background, var(--p-primary-color, #4a8cd0))";
const _BTN_PRIMARY_FG = "var(--p-button-primary-color, #fff)";
const _BTN_PRIMARY_BORDER = "var(--p-button-primary-border-color, var(--p-primary-color, #4a8cd0))";

function _overlay() {
    const el = document.createElement("div");
    el.style.cssText = `
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.55);
        z-index: 10000;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--p-font-family, system-ui, -apple-system, "Segoe UI", sans-serif);
    `;
    return el;
}
function _dialog(minWidth = 360) {
    const d = document.createElement("div");
    d.style.cssText = `
        background: ${_DIALOG_BG}; color: ${_DIALOG_FG};
        border: 1px solid ${_DIALOG_BORDER};
        border-radius: 6px; padding: 18px 20px 16px;
        min-width: ${minWidth}px; max-width: 520px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    `;
    return d;
}
function _btn(label, primary = false, danger = false) {
    const b = document.createElement("button");
    b.textContent = label;
    let bg = _BTN_BG, fg = _BTN_FG, bd = _BTN_BORDER;
    if (primary) { bg = _BTN_PRIMARY_BG; fg = _BTN_PRIMARY_FG; bd = _BTN_PRIMARY_BORDER; }
    if (danger) {
        bg = "var(--p-button-danger-background, #c44)";
        fg = "var(--p-button-danger-color, #fff)";
        bd = "var(--p-button-danger-border-color, #c44)";
    }
    b.style.cssText = `
        background: ${bg}; color: ${fg}; border: 1px solid ${bd};
        padding: 6px 14px; border-radius: 4px; cursor: pointer;
        font-size: 13px; font-family: inherit; min-width: 72px;
    `;
    b.onmouseenter = () => { b.style.opacity = "0.85"; };
    b.onmouseleave = () => { b.style.opacity = "1"; };
    return b;
}

function showThemedPrompt({ title, message, defaultValue = "", placeholder = "",
                            okLabel = "Save", cancelLabel = "Cancel" }) {
    return new Promise((resolve) => {
        const ov = _overlay(); const dlg = _dialog(380);
        if (title) {
            const t = document.createElement("div");
            t.textContent = title;
            t.style.cssText = "font-weight:600;font-size:14px;margin-bottom:10px;";
            dlg.appendChild(t);
        }
        if (message) {
            const m = document.createElement("div");
            m.textContent = message;
            m.style.cssText = "font-size:12px;line-height:1.5;opacity:0.85;margin-bottom:12px;white-space:pre-wrap;";
            dlg.appendChild(m);
        }
        const inp = document.createElement("input");
        inp.type = "text"; inp.value = defaultValue; inp.placeholder = placeholder;
        inp.style.cssText = `
            width: 100%; box-sizing: border-box;
            background: ${_INPUT_BG}; color: ${_INPUT_FG};
            border: 1px solid ${_INPUT_BORDER};
            padding: 7px 9px; border-radius: 4px; font-size: 13px;
            font-family: inherit; outline: none;
        `;
        dlg.appendChild(inp);
        const row = document.createElement("div");
        row.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:14px;";
        const cb = _btn(cancelLabel, false), ob = _btn(okLabel, true);
        row.appendChild(cb); row.appendChild(ob); dlg.appendChild(row);
        ov.appendChild(dlg); document.body.appendChild(ov);
        setTimeout(() => { inp.focus(); inp.select(); }, 0);
        const close = (v) => { ov.parentNode && ov.parentNode.removeChild(ov); resolve(v); };
        cb.onclick = () => close(null);
        ob.onclick = () => close(inp.value);
        ov.onclick = (e) => { if (e.target === ov) close(null); };
        inp.onkeydown = (e) => {
            if (e.key === "Enter") { e.preventDefault(); close(inp.value); }
            if (e.key === "Escape") { e.preventDefault(); close(null); }
        };
    });
}

function showThemedConfirm({ title, message, okLabel = "OK",
                              cancelLabel = "Cancel", danger = false }) {
    return new Promise((resolve) => {
        const ov = _overlay(); const dlg = _dialog(340);
        if (title) {
            const t = document.createElement("div");
            t.textContent = title;
            t.style.cssText = "font-weight:600;font-size:14px;margin-bottom:10px;";
            dlg.appendChild(t);
        }
        const m = document.createElement("div");
        m.textContent = message || "";
        m.style.cssText = "font-size:13px;line-height:1.5;white-space:pre-wrap;";
        dlg.appendChild(m);
        const row = document.createElement("div");
        row.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:16px;";
        const cb = _btn(cancelLabel, false), ob = _btn(okLabel, !danger, danger);
        row.appendChild(cb); row.appendChild(ob); dlg.appendChild(row);
        ov.appendChild(dlg); document.body.appendChild(ov);
        setTimeout(() => ob.focus(), 0);
        const close = (v) => { ov.parentNode && ov.parentNode.removeChild(ov); resolve(v); };
        cb.onclick = () => close(false); ob.onclick = () => close(true);
        ov.onclick = (e) => { if (e.target === ov) close(false); };
        document.addEventListener("keydown", function onKey(e) {
            if (!ov.parentNode) { document.removeEventListener("keydown", onKey, true); return; }
            if (e.key === "Escape") { e.preventDefault(); close(false); }
            if (e.key === "Enter")  { e.preventDefault(); close(true); }
        }, true);
    });
}

function showToast(message, kind = "info", duration = 3200) {
    const t = document.createElement("div");
    t.textContent = message;
    const accent =
        kind === "error"   ? "var(--p-button-danger-background, #c44)" :
        kind === "success" ? "var(--p-button-success-background, #5a4)" :
        kind === "warn"    ? "var(--p-button-warn-background, #d80)" :
                             "var(--p-primary-color, #4a8cd0)";
    t.style.cssText = `
        position: fixed; bottom: 24px; right: 24px;
        background: ${_DIALOG_BG}; color: ${_DIALOG_FG};
        border: 1px solid ${_DIALOG_BORDER};
        border-left: 4px solid ${accent};
        padding: 10px 16px; border-radius: 4px;
        z-index: 10001; font-size: 13px; max-width: 380px;
        font-family: var(--p-font-family, system-ui, sans-serif);
        box-shadow: 0 4px 14px rgba(0,0,0,0.45);
        white-space: pre-wrap; line-height: 1.4;
        opacity: 0; transform: translateY(8px);
        transition: opacity 0.18s, transform 0.18s;
    `;
    document.body.appendChild(t);
    requestAnimationFrame(() => {
        t.style.opacity = "1"; t.style.transform = "translateY(0)";
    });
    setTimeout(() => {
        t.style.opacity = "0"; t.style.transform = "translateY(8px)";
        setTimeout(() => t.parentNode && t.parentNode.removeChild(t), 250);
    }, duration);
}

// ----- Server helpers -------------------------------------------------

async function refreshNodeDefs() {
    try {
        const cmd = app?.extensionManager?.command;
        if (cmd && typeof cmd.execute === "function") {
            await cmd.execute("Comfy.RefreshNodeDefinitions");
        }
    } catch (e) {
        console.warn("[ManualSampler] RefreshNodeDefinitions failed:", e);
    }
}

async function fetchSamplerInfo(name) {
    try {
        const r = await fetch(
            `/RES4SHO/manual_sampler/sampler_info?name=${encodeURIComponent(name)}`);
        if (!r.ok) return { params: [], accepts_var_kwargs: false };
        return await r.json();
    } catch (e) {
        console.warn("[ManualSampler] sampler_info fetch failed:", e);
        return { params: [], accepts_var_kwargs: false };
    }
}

async function listPresets() {
    try {
        const r = await fetch("/RES4SHO/manual_sampler/presets");
        if (!r.ok) throw new Error("HTTP " + r.status);
        return await r.json();
    } catch (e) {
        console.warn("[ManualSampler] listPresets failed:", e);
        return { presets: {}, prefix: "manual_sampler_" };
    }
}

async function savePreset(name, base_sampler, eta_override, s_noise, stages) {
    const r = await fetch("/RES4SHO/manual_sampler/preset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            name, base_sampler, eta_override, s_noise, stages,
        }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
    return data;
}

async function deletePreset(name) {
    const r = await fetch(
        `/RES4SHO/manual_sampler/preset?name=${encodeURIComponent(name)}`,
        { method: "DELETE" });
    const data = await r.json().catch(() => ({}));
    return !!data.ok;
}

let _lastInteractionEvent = null;
if (typeof window !== "undefined") {
    document.addEventListener("mousedown", (e) => { _lastInteractionEvent = e; }, true);
    document.addEventListener("pointerdown", (e) => { _lastInteractionEvent = e; }, true);
}
function _evtFromMouse() {
    const lm = app?.canvas?.last_mouse;
    if (Array.isArray(lm) && lm.length >= 2) {
        return { clientX: lm[0], clientY: lm[1] };
    }
    return { clientX: 200, clientY: 200 };
}

// ----- Extension registration -----------------------------------------

app.registerExtension({
    name: "RES4SHO.ManualSampler",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ManualSampler") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const node = this;
            const baseW   = this.widgets?.find(w => w.name === "base_sampler");
            const stagesW = this.widgets?.find(w => w.name === "stages");
            const etaW    = this.widgets?.find(w => w.name === "eta_override");
            const snW     = this.widgets?.find(w => w.name === "s_noise");
            const dataW   = this.widgets?.find(w => w.name === "preset_data");

            // Hide preset_data; it's edit-via-frontend storage only.
            // Coerce its value to a string and override serializeValue
            // so that third-party extensions hooking string widgets (e.g.
            // the presetText extension which calls value.replace(...)) can
            // never receive null and crash graphToPrompt.
            if (dataW) {
                dataW.type = "hidden";
                dataW.computeSize = () => [0, -4];
                if (dataW.value == null) dataW.value = "";
                const origSerialize = dataW.serializeValue;
                dataW.serializeValue = function () {
                    const v = (typeof origSerialize === "function")
                        ? origSerialize.apply(this, arguments) : this.value;
                    return v == null ? "" : v;
                };
            }

            // Toggle a number widget's visibility by collapsing its row.
            // Use an explicit state flag because some widgets default
            // to ``computeSize === undefined`` (LiteGraph supplies the
            // default). A naive "restore only if the saved value isn't
            // undefined" guard leaves the widget permanently collapsed
            // after the first hide/show cycle.
            function setHidden(widget, hidden) {
                if (!widget) return;
                if (hidden) {
                    if (widget._ms_hidden_state !== "hidden") {
                        widget._ms_origType = widget.type;
                        widget._ms_origComputeSize = widget.computeSize;
                        widget._ms_hidden_state = "hidden";
                    }
                    widget.type = "hidden";
                    widget.computeSize = () => [0, -4];
                } else {
                    if (widget._ms_hidden_state === "hidden") {
                        widget.type = widget._ms_origType;
                        widget.computeSize = widget._ms_origComputeSize;
                        widget._ms_hidden_state = "shown";
                    }
                }
            }

            async function refreshSamplerInfo() {
                if (!baseW || !baseW.value) return;
                const info = await fetchSamplerInfo(baseW.value);
                const params = (info.params || []).map(p => p.name);
                const paramSet = new Set(params);

                const acceptsEta    = paramSet.has("eta")    || info.accepts_var_kwargs;
                const acceptsSn     = paramSet.has("s_noise") || info.accepts_var_kwargs;
                const acceptsStages = paramSet.has("stages")  || info.accepts_var_kwargs;
                setHidden(stagesW, !acceptsStages);
                setHidden(etaW, !acceptsEta);
                setHidden(snW, !acceptsSn);

                // Surface the accepted-params info via the base_sampler
                // widget's tooltip + a console line so the user can see
                // what the chosen sampler actually exposes (including
                // non-standard knobs like eta_peak / s_churn / r that we
                // can't bind to a widget without per-sampler UI).
                const standard = new Set(["eta", "s_noise", "stages"]);
                const others = params.filter(n => !standard.has(n));
                const widgetParts = [];
                if (acceptsStages) widgetParts.push("stages");
                if (acceptsEta)    widgetParts.push("eta");
                if (acceptsSn)     widgetParts.push("s_noise");
                let summary;
                if (info.accepts_var_kwargs) {
                    summary = "accepts **kwargs (any)";
                } else if (params.length === 0) {
                    summary = "no tunable knobs";
                } else {
                    summary = "tunable: " + params.join(", ");
                }
                if (baseW) {
                    baseW.options = Object.assign(baseW.options || {}, {
                        tooltip: `${baseW.value}: ${summary}`,
                    });
                }
                console.debug(
                    `[ManualSampler] ${baseW.value}: `,
                    `widget-bound=[${widgetParts.join(", ") || "none"}], `,
                    `other params=[${others.join(", ") || "none"}], `,
                    `var_kwargs=${!!info.accepts_var_kwargs}`);
                node.setSize?.(node.computeSize?.() || node.size);
                node.setDirtyCanvas?.(true, true);
            }

            // Wrap the base_sampler combo's callback so that picking a
            // new sampler refreshes the info / visibility.
            if (baseW) {
                const orig = baseW.callback;
                baseW.callback = function (...args) {
                    const ret = orig?.apply(this, args);
                    refreshSamplerInfo();
                    return ret;
                };
                // Initial paint
                refreshSamplerInfo();
            }

            this.addWidget(
                "button", "save preset…", null,
                async () => {
                    if (!baseW) return;
                    const name = await showThemedPrompt({
                        title: "Save manual sampler preset",
                        message: "Allowed: letters, digits, spaces, dashes, "
                            + "underscores (max 64 chars). The preset will appear "
                            + "in every sampler dropdown as "
                            + "\"manual_sampler_<name>\" once node definitions "
                            + "refresh.",
                        placeholder: "e.g. my_dpmpp_eta",
                        okLabel: "Save",
                    });
                    if (name === null) return;
                    const trimmed = String(name).trim();
                    if (!trimmed) return;
                    try {
                        const eta = (etaW?.value ?? -1);
                        // Only persist stages / s_noise if the chosen sampler
                        // accepts them -- otherwise the saved preset would
                        // carry an inert value that misleads on reload.
                        // We rely on the type the widget is currently set to
                        // (set by setHidden) so the check is robust even if
                        // the widget never had its hidden state cycled.
                        const stagesAccepted = stagesW && stagesW.type !== "hidden";
                        const snAccepted     = snW    && snW.type    !== "hidden";
                        const result = await savePreset(
                            trimmed,
                            baseW.value,
                            eta < 0 ? null : eta,
                            snAccepted ? (snW.value ?? null) : null,
                            stagesAccepted ? (stagesW.value ?? null) : null,
                        );
                        await refreshNodeDefs();
                        showToast(
                            `Saved "${trimmed}".\n`
                            + `Available as "${result.sampler || ("manual_sampler_" + trimmed)}" `
                            + "in sampler dropdowns.",
                            "success");
                    } catch (e) {
                        showToast("Save failed: " + e.message, "error");
                    }
                },
                { serialize: false },
            );

            this.addWidget(
                "button", "load preset…", null,
                async () => {
                    const data = await listPresets();
                    const names = Object.keys(data.presets || {});
                    if (!names.length) {
                        showToast(
                            "No saved presets yet. Configure a sampler "
                            + "and use \"save preset…\" first.", "info");
                        return;
                    }
                    const evt = _lastInteractionEvent || _evtFromMouse();
                    new LiteGraph.ContextMenu(names, {
                        event: evt,
                        callback: (selected) => {
                            const p = data.presets[selected];
                            if (!p) return;
                            if (baseW && p.base_sampler) baseW.value = p.base_sampler;
                            if (stagesW && p.stages != null) stagesW.value = p.stages;
                            if (etaW)  etaW.value  = (p.eta_override == null) ? -1 : p.eta_override;
                            if (snW)   snW.value   = (p.s_noise == null) ? 1.0 : p.s_noise;
                            if (dataW) dataW.value = JSON.stringify({
                                loaded_from: selected,
                                base_sampler: p.base_sampler,
                                stages: p.stages,
                                eta_override: p.eta_override,
                                s_noise: p.s_noise,
                            });
                            // Fire the wrapped callback so visibility
                            // toggles for the loaded sampler.
                            if (baseW && typeof baseW.callback === "function") {
                                baseW.callback(baseW.value, app.canvas, this);
                            }
                            this.setDirtyCanvas?.(true, true);
                            showToast(`Loaded "${selected}".`, "success", 2000);
                        },
                    });
                },
                { serialize: false },
            );

            this.addWidget(
                "button", "delete preset…", null,
                async () => {
                    const data = await listPresets();
                    const names = Object.keys(data.presets || {});
                    if (!names.length) {
                        showToast("No saved presets to delete.", "info");
                        return;
                    }
                    const evt = _lastInteractionEvent || _evtFromMouse();
                    new LiteGraph.ContextMenu(names, {
                        event: evt,
                        callback: async (selected) => {
                            const ok = await showThemedConfirm({
                                title: "Delete preset?",
                                message: `Delete preset "${selected}"?\n\n`
                                    + "This also unregisters its "
                                    + `"manual_sampler_${selected}" sampler.`,
                                okLabel: "Delete",
                                danger: true,
                            });
                            if (!ok) return;
                            const success = await deletePreset(selected);
                            if (success) {
                                await refreshNodeDefs();
                                showToast(`Deleted "${selected}".`, "success", 2000);
                            } else {
                                showToast("Delete failed.", "error");
                            }
                        },
                    });
                },
                { serialize: false },
            );

            return r;
        };
    },
});
