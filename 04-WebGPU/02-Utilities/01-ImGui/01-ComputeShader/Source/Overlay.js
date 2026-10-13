import { ImGui, ImGuiImplWeb, ImGuiImplWGPU, ImVec2, ImGuiCond, ImGuiWindowFlags, ImGuiTreeNodeFlags } from "./jsimgui/imgui.js";

const HISTORY_SIZE = 120;
const SMOOTHING_TIME_CONSTANT = 0.5;
const HISTORY_SAMPLE_INTERVAL = 1.0 / 60.0;

const stats = 
{
    lastTime: 0.0,
    frameTime: 0.0,
    smoothedFrameTime: 0.0,
    currentFPS: 0.0,
    avgFPS: 0.0,
    minFPS: Number.MAX_VALUE,
    maxFPS: 0.0,
    frameCount: 0,
    accumulatedTime: 0.0,
    history: new Array(HISTORY_SIZE).fill(0.0),
    historyOffset: 0,
    historySampleAccumulator: 0.0 
};

let overlayCanvas = null;
let bInitialized = false;
let bShowPerformanceStats = false;
let gpuDescription = "";

const categories = new Map();

function addItem(_categoryName, _drawFunc)
{
    // Code
    if (!categories.has(_categoryName))
        categories.set(_categoryName, []);

    categories.get(_categoryName).push(_drawFunc);
}

function drawPerformanceStats()
{
    // Code
    ImGui.Text(`GPU : ${gpuDescription}`);
    ImGui.Text(`Resolution : ${overlayCanvas.width} x ${overlayCanvas.height}`);
    ImGui.Separator();
    ImGui.Text(`FPS : ${stats.currentFPS.toFixed(1)}`);
    ImGui.Text(`Frame Time : ${stats.smoothedFrameTime.toFixed(3)} ms`);
    ImGui.Text(`Min / Max / Avg FPS : ${stats.frameCount > 0 ? stats.minFPS.toFixed(1) : "0.0"} / ${stats.maxFPS.toFixed(1)} / ${stats.avgFPS.toFixed(1)}`);
    ImGui.PlotLines("##FPS", stats.history, HISTORY_SIZE, stats.historyOffset, "FPS", 0.0, Math.max(stats.maxFPS * 1.2, 60.0), new ImVec2(300.0, 80.0));
}

const Overlay = 
{
    async Init(_canvas, _device, _adapterInfo, _fontPath, _fontSize)
    {
        // Code
        overlayCanvas = _canvas;

        await ImGuiImplWeb.Init({ 
            canvas: _canvas, 
            device: _device, 
            backend: "webgpu"
        });
        ImGui.StyleColorsDark();

        if (_fontPath)
        {
            const response = await fetch(_fontPath)
            if (response.ok)
            {
                ImGuiImplWeb.LoadFont("Overlay.ttf", new Uint8Array(await response.arrayBuffer()));
                ImGui.GetIO().FontDefault = ImGui.GetIO().Fonts.AddFontFromFileTTF("Overlay.ttf", _fontSize);
            }
            else
                console.log(`Overlay : Failed To Load Font ${_fontPath} ... Fallback To Default Font !!!`);
        }

        gpuDescription = _adapterInfo.description || `${_adapterInfo.vendor} ${_adapterInfo.architecture}`.trim() || "Unknown";

        window.addEventListener("focus", Overlay.ResetPerformanceStats);
        document.addEventListener("visibilitychange", Overlay.ResetPerformanceStats);

        bInitialized = true;
        console.log("Overlay Initialized Successfully");
    },

    Cleanup()
    {
        // Code
        if (!bInitialized)
            return;

        window.removeEventListener("focus", Overlay.ResetPerformanceStats);
        document.removeEventListener("visibilitychange", Overlay.ResetPerformanceStats);

        ImGuiImplWGPU.Shutdown();
        ImGui.DestroyContext();
        categories.clear();
        bInitialized = false;
    },

    WantCaptureKeyboard()
    {
        return bInitialized && ImGui.GetIO().WantCaptureKeyboard;
    },

    //! Performance Stats
    ShowPerformanceStats()
    {
        bShowPerformanceStats = true;
    },

    UpdatePerformanceStats()
    {
        // Code
        const now = performance.now();
        if (stats.lastTime == 0.0)
        {
            stats.lastTime = now;
            return;
        }

        const deltaTime = (now - stats.lastTime) / 1000.0;
        stats.lastTime = now;
        if (deltaTime <= 0.0)
            return;

        stats.frameTime = deltaTime * 1000.0;

        const alpha = 1.0 - Math.exp(-deltaTime / SMOOTHING_TIME_CONSTANT);
        if (stats.smoothedFrameTime == 0.0)
            stats.smoothedFrameTime = stats.frameTime;
        else
            stats.smoothedFrameTime += alpha * (stats.frameTime - stats.smoothedFrameTime);

        stats.currentFPS = 1000.0 / stats.smoothedFrameTime;

        stats.frameCount++;

        stats.accumulatedTime += deltaTime;
        stats.avgFPS = stats.frameCount / stats.accumulatedTime;
        stats.minFPS = Math.min(stats.minFPS, stats.currentFPS);
        stats.maxFPS = Math.max(stats.maxFPS, stats.currentFPS);

        stats.historySampleAccumulator += deltaTime;
        if (stats.historySampleAccumulator >= HISTORY_SAMPLE_INTERVAL)
        {
            stats.history[stats.historyOffset] = stats.currentFPS;
            stats.historyOffset = (stats.historyOffset + 1) % HISTORY_SIZE;
            stats.historySampleAccumulator %= HISTORY_SAMPLE_INTERVAL;
        }
    },

    ResetPerformanceStats()
    {
        // Code
        stats.lastTime = 0.0;
        stats.frameTime = 0.0;
        stats.smoothedFrameTime = 0.0;
        stats.currentFPS = 0.0;
        stats.avgFPS = 0.0;
        stats.minFPS = Number.MAX_VALUE;
        stats.maxFPS = 0.0;
        stats.frameCount = 0;
        stats.accumulatedTime = 0.0;
        stats.history.fill(0.0);
        stats.historyOffset = 0;
        stats.historySampleAccumulator = 0.0;
    },

    //! Widgets
    AddText(_categoryName, _text)
    {
        // Code
        addItem(_categoryName, () => ImGui.Text(_text));
    },

    AddDynamicText(_categoryName, _label, _valueFunc)
    {
        // Code
        addItem(_categoryName, () => ImGui.Text(`${_label} : ${_valueFunc()}`));
    },

    AddRadioButton(_categoryName, _label, _isActiveFunc, _callback, _sameLine = false)
    {
        // Code
        addItem(_categoryName, () => 
        {
            if (_sameLine)
                ImGui.SameLine();
            if (ImGui.RadioButton(_label, _isActiveFunc()))
                _callback();
        });
    },

    AddColorEdit3(_categoryName, _label, _colorArray, _callback = null)
    {
        // Code
        addItem(_categoryName, () =>
        {
            const rgb = [_colorArray[0], _colorArray[1], _colorArray[2]];
            if (ImGui.ColorEdit3(_label, rgb))
            {
                _colorArray[0] = rgb[0];
                _colorArray[1] = rgb[1];
                _colorArray[2] = rgb[2];
                if (_callback)
                    _callback();
            }
        });
    },

    AddSpacing(_categoryName)
    {
        // Code
        addItem(_categoryName, () => ImGui.Separator());
    },

    NewFrame()
    {
        // Code
        if (!bInitialized)
            return;

        ImGuiImplWeb.BeginRender();

        ImGui.SetNextWindowPos(new ImVec2(380.0, 520.0), ImGuiCond.FirstUseEver);
        ImGui.Begin("WebGPU : ImGui Overlay");
        {
            if (bShowPerformanceStats && ImGui.CollapsingHeader("Performance", ImGuiTreeNodeFlags.DefaultOpen))
                drawPerformanceStats();

            for (const [categoryName, items] of categories)
            {
                if (!ImGui.CollapsingHeader(categoryName, ImGuiTreeNodeFlags.DefaultOpen))
                    continue;

                for (const drawFunc of items)
                    drawFunc();
            }
        }   
        ImGui.End();     
    },

    Render(passEncoder)
    {
        // Code
        if (!bInitialized)
            return;

        ImGuiImplWeb.EndRender(passEncoder);
    }
};

//* Expose to Canvas.js
window.Overlay = Overlay;


