const fInput = document.getElementById('fileInput');
const optionSelect = document.querySelectorAll('.valSelect');

const section = {}
section.compPanel = document.getElementById('comparison-panel');
section.dropZone = document.getElementById('drop-zone');
section.convert = document.getElementById('convert-buttons');
section.upscale = document.querySelector('.upscaler-buttons');
section.loading = document.getElementById('loading');

let Output = null;
const inferenceOptions = {};
var Models = {};

optionSelect.forEach(el => {
    el.value = el.options[Number(el.dataset.s)].value;
    const event = new Event('change');
    el.dispatchEvent(event);
})

wnx.meta.providers.forEach((e) => {
    const opt = document.createElement('option');
    opt.textContent = e;
    opt.value = e;
    optionSelect[1].appendChild(opt);
});

// view states

function sectionStates(states = {}) {
    for (const elem in states) {
        section[elem].style.display = states[elem];
    }
}

sectionStates({ dropZone: 'flex', compPanel: 'none', convert: 'none', loading: 'none' });

async function setModelList() {
    Models = await wnx.meta.requestModelsCollections('/get-models-info');
    delete Models['readme.md'];
    delete Models['.gitignore'];
    for (const name in Models) {
        const opt = document.createElement('option');
        opt.textContent = name;
        opt.value = name;
        optionSelect[0].appendChild(opt);
    }
    if (Models['Real-ESRGAN-General-x4v3']){
        Models['Real-ESRGAN-General-x4v3'].tileSize = 128;
    }
}

setModelList();
// #region comparison panel

const pv = {};
pv.svgSlider = document.querySelector('.draggable-svg');
pv.compContainer = document.querySelector('.comparison-container');
pv.afterImgParent = document.querySelector('.image-left');
pv.beforeImg = document.getElementById("beforeImg");
pv.afterImg = document.getElementById("afterImg");
pv.resModel = document.getElementById("ai-model-value");
pv.resMulti = document.getElementById("ai-multi-value");
pv.resOriginSize = document.getElementById("ai-origin-value");
pv.startBtn = document.getElementById('startInterference');

pv.afterImg.setAttribute('draggable', false);
pv.beforeImg.setAttribute('draggable', false);
var isDragging = false;

// comparison slider drag handler

pv.svgSlider.addEventListener("mousedown", (e) => {
    e.preventDefault();
    isDragging = true;
    startX = e.clientX - pv.svgSlider.offsetLeft;
    pv.svgSlider.style.cursor = "grabbing";
});

document.addEventListener("mousemove", (e) => {
    if (isDragging) {
        let newX = e.clientX - startX;

        // Clamp the newX position within container boundaries
        newX = Math.max(0, Math.min(newX, pv.compContainer.offsetWidth));

        // Set the new X position of the SVG
        pv.svgSlider.style.left = newX + "px";
        pv.afterImgParent.style.width = newX + "px";
    }
});

document.addEventListener("mouseup", () => {
    isDragging = false;
    pv.svgSlider.style.cursor = "grab";
});

// Touch Events
pv.svgSlider.addEventListener("touchstart", (e) => {
    e.preventDefault();
    isDragging = true;
    const touch = e.touches[0]; // Get the first touch point
    startX = touch.clientX - pv.svgSlider.offsetLeft;
    pv.svgSlider.style.cursor = "grabbing";
});

// Touch Move Event
document.addEventListener("touchmove", (e) => {
    if (isDragging) {
        const touch = e.touches[0]; // Get the first touch point
        let newX = touch.clientX - startX;

        // Clamp the newX position within container boundaries
        newX = Math.max(0, Math.min(newX, pv.compContainer.offsetWidth - pv.svgSlider.offsetWidth));

        // Set the new X position of the SVG
        pv.svgSlider.style.left = newX + "px";
        pv.afterImgParent.style.width = newX + "px";
    }
});

// Touch End Event
document.addEventListener("touchend", () => {
    isDragging = false;
    pv.svgSlider.style.cursor = "grab";
});


// zoom in/out on comparison panel

let scale = 1;
pv.maxScale = 3;

const zoomFactor = 0.5;

pv.compContainer.addEventListener('wheel', (event) => {
    scale += event.deltaY * -zoomFactor;

    // Clamp scale to achieve width between 100px (0.17x) and 300px (0.5x)
    scale = Math.min(Math.max(1, scale), 4);

    pv.beforeImg.style.transform = `scale(${scale})`;
    pv.afterImg.style.transform = `scale(${scale})`;
}, { passive: false });

let initialDistance = null;

pv.compContainer.addEventListener('touchmove', (event) => {
    if (event.touches.length === 2) {
        const distX = event.touches[0].clientX - event.touches[1].clientX;
        const distY = event.touches[0].clientY - event.touches[1].clientY;
        const currentDistance = Math.sqrt(distX * distX + distY * distY);

        if (!initialDistance) initialDistance = currentDistance;

        scale *= currentDistance / initialDistance;
        scale = Math.min(Math.max(1, scale), 4);

        pv.beforeImg.style.transform = `scale(${scale})`;;
        pv.afterImg.style.transform = `scale(${scale})`;;

        initialDistance = currentDistance;
    }
}, { passive: false });

pv.compContainer.addEventListener('touchend', () => {
    initialDistance = null;
});

//#endregion
//#region image pre + processing  

document.querySelector('.close-button').addEventListener('click', (e) => {
    sectionStates({ dropZone: 'flex', compPanel: 'none', upscale: 'flex', convert: 'none', loading: 'none' });
    pv.startBtn.style.display = 'none';
    fInput.value = '';
})

section.dropZone.addEventListener('click', () => { fInput.click() });

pv.startBtn.style.display = 'none';
wnx.handlers.chunkProcess.doneEvent = () => {
    wnx.handlers.chunkProcess.total -= 1;
    section.loading.querySelector('button').textContent = `Inference is running... remaining chunks: ${wnx.handlers.chunkProcess.total}`;
}

document.getElementById('encodes').addEventListener('click', (e) => {
    const format = optionSelect[2].value;
    const quality = document.getElementById('output').textContent + 0;
    wnx.Image.encodeRGBA(Output.imageData.data, Output.imageData.width, Output.imageData.height, quality, format).then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `upscaled_image.${format}`;
        a.click();
        URL.revokeObjectURL(url);
    });
});


fInput.addEventListener('change', async (e) => {
    try {
        const file = e.target.files[0]; // Get the selected file
        const reader = new FileReader();
        reader.onload = function (event) {
            const blob = new Blob([reader.result], { type: file.type });
            const objectURL = URL.createObjectURL(blob);
            pv.beforeImg.src = objectURL;
            pv.startBtn.style.display = 'block';
            sectionStates({ dropZone: 'none', compPanel: 'flex' });
            pv.afterImg.src = '';
        }
        reader.readAsArrayBuffer(file);
    }
    catch (error) {
        console.error(error);
    }
});

pv.startBtn.addEventListener('click', async function () {
    try {
        pv.startBtn.style.display = "none";
        sectionStates({ upscale: 'none', loading: 'flex' });

        wnx.env.logLevel = "verbose";
        wnx.cfg.wasmGpuRunOnWorker = optionSelect[3].value == "true";
        wnx.cfg.avgChunkSize = Number(optionSelect[4].value);
        wnx.setWasmFlags({ proxy: false, numThreads: 8 });
        const provider = optionSelect[1].value == 'auto' ? ['webgpu', 'wasm'] : [optionSelect[1].value];
        inferenceOptions.executionProviders = provider;
        wnx.setInferenceOption(inferenceOptions);

        const onnxModel = new wnx.Model(Models[optionSelect[0].value]);
        Output = new wnx.OutputData({ includeTensor: false, dumpOriginalImageData: true });
        await wnx.inferenceRun(onnxModel, fInput.files[0], Output);

        console.log('Inference finished', Output.imageData.data.length, '- Time to finish:', Output.time);

        const imgsrc = await wnx.Image.imgUrlFromRGB(Output.imageData.data, Output.imageData.width, Output.imageData.height);
        pv.afterImg.src = imgsrc;

        pv.resOriginSize.textContent = `${wnx.d_in.w}x${wnx.d_in.h}`;
        document.getElementById('ai-model-value').textContent = `${Output.time} seconds`;
        document.getElementById('ai-multi-value').textContent = ` ${Output.multiplier}`
        document.querySelectorAll('.image-wrapper')[0].style.width = '50%';
        sectionStates({ upscale: 'none', convert: 'flex', loading: 'none' });
    }
    catch (e) {
        alert(e);
        console.error(e);
        sectionStates({ dropZone: 'flex', compPanel: 'none', upscale: 'flex', convert: 'none', loading: 'none' });
    }
});