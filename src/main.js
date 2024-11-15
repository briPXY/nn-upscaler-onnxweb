// import * as ort from 'onnxruntime-web';  // Bundling not includes webgpu backend despite stated true.
import * as Image from './image_helper';
import * as meta from './meta';

// worker raw
import sessionRawWorker from './webgpu_session.worker';

export const cfg = {
    webGpuBufferLocation: 'cpu',
    wasmGpuRunOnWorker: true, // true = run inference session on worker (not available for webgl)
    backendPath: {
        webgpu: `${meta._defaultModulePath}ort.webgpu.min.js`,
        wasm: `${meta._defaultModulePath}ort.wasm.min.js`,
        all: `${meta._defaultModulePath}ort.all.min.js`
    }
};

// To be set on ort.env.
export const _env = {
    wasm: {},
};

_env.logLevel = 'info';
_env.wasm.wasmPaths = `${meta._defaultModulePath}`;
_env.wasm.proxy = false;
_env.wasm.numThreads = meta.threads.default;

export const InferenceOpt = {
    executionProviders: meta.providers,
};

// Input data
export const d_in = {
    tensor: [],
    dims: [1, 3, 244, 244],
    c: 0,
    w: 0,
    h: 0,
};

export const d_out = {
    imageData: [],
    tensor: [],
    multiplier: null, //  result dimension / original dimension
};

// Load backend for main-thread inference.
export async function loadBackendScript(provider = 'all', id='onnx-backend') {
    return new Promise((resolve, reject) => {
        const script = document.getElementById(id);

        if (script.src != ''){
            resolve('loaded');
            return;
        }

        if (!script){
            console.error(`'No script tag with id="${id}" is provided'`);
            reject(`No script tag with id="${id}" provided`);
        }

        script.src = cfg.backendPath[provider];
        script.onload = () => {
            for (const key in _env) {
                if (typeof _env[key] !== 'object') {
                    ort.env[key] = _env[key];
                }
            }

            for (const key in _env.wasm) {
                ort.env.wasm[key] = _env.wasm[key];
            }

            if (_env.webgpu) {
                for (const key in _env.webgpu) {
                    ort.env.webgpu[key] = _env.webgpu[key];
                }
            } 
            resolve('loaded');
        };
        script.onerror = (error) => {
            console.error(`loadRuntimeBackend() error @ loading ${provider} backend`);
            reject(error);
        };
    }); 
}

// future implementation for IndexedDB model storing
const _ModelCollections = {};

// This server use only. Return models info from all model stored in static folder in ordered directory (/channel/datatype/dims/modelfile.onnx).
// Useful for automating list element.
export async function requestModelsCollections(api_url) {
    try {
        const response = await fetch(api_url);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const modelPaths = await response.json();

        modelPaths.forEach(url => {
            const pathText = url.match(/[a-zA-Z0-9\.\_\-]+/g);
            const l = pathText.length;
            const name = pathText[l - 1].replace('.onnx', '');
            _ModelCollections[name] = {
                channel: parseInt(pathText[l - 4]),
                dataType: pathText[l - 3],
                layout: pathText[l - 2].toUpperCase(),// tensor format
                url: meta._domain + url.replace(/static|\\/g, '/').replace('//', '/'),
                rawPath: url,
            };
        });

        return _ModelCollections;
    } catch (error) {
        console.error('Error fetching filenames:', error);
    }
}

export function setWasmFlags(flags = {}) {
    for (const flag in flags) {
        _env.wasm[flag] = flags[flag];
    }
    return;
}

export function setWebGPUFlags(flags = {}) {
    for (const flag in flags) {
        _env.webgpu[flag] = flags[flag];
    }
    return;
}

export function setEnvFlag(prop = 'string', value) {
    _env[prop] = value;
    return;
}

export function setInferenceOption(obj = {}) {
    for (const prop in obj) {
        InferenceOpt[prop] = obj[prop];
    }
    return;
}

// Set path for onnxweb runtime distributions. Will override all runtime provider path.
export function setRuntimePath(url) {
    /\/$/.test(url) ? url : url += '/';
    _env.wasm.wasmPaths = url;
    cfg.backendPath.webgpu = `${url}ort.webgpu.min.js`;
    cfg.backendPath.wasm = `${url}ort.wasm.min.js`;
}

// Get model info into ModelInfo. Using formatted URL. 
// Path should in an format of '.../channel/dataType/tensorFormat/model-name.onnx', URL must ended with .onnx.
export async function makeModelInfo_formatted(modelUrl) {
    try {
        const response = await meta._checkUrlExists(modelUrl);

        if (!response) {
            throw new Error(`Model URL ${modelUrl} not exit!`);
        }

        const ModelInfo = {}
        const pathText = modelUrl.match(/[a-zA-Z0-9\.\_\-]+/g);
        const l = pathText.length;
        const name = pathText[l - 1].replace('.onnx', '');
        ModelInfo.name = name;
        ModelInfo.channel = parseInt(pathText[l - 4]);
        ModelInfo.dataType = pathText[l - 3];
        ModelInfo.layout = pathText[l - 2].toUpperCase();
        ModelInfo.url = modelUrl;

        return ModelInfo;
    } catch (error) {
        console.error('Error fetching url:', error);
    }
}

// If using CDN or self-hosted, get model info manually, pass the modelUrl with string of URL ended with .onnx
export async function makeModelInfo(modelUrl, dataType, layout, channel = 3) {
    const urlCheck = await meta._checkUrlExists(modelUrl)

    if (!urlCheck) {
        throw 'Model url not exist';
    }

    const ModelInfo = {};
    const name = modelUrl.match(/[^\/]+\.onnx/i)[0];
    ModelInfo.name = name;
    ModelInfo.channel = channel;
    ModelInfo.dataType = dataType; // fp32, fp16 etc
    ModelInfo.url = modelUrl;
    ModelInfo.layout = layout.toUpperCase(); // nchw/nhwc
    return ModelInfo;
}


const _setWebGpuExecProvider = (layout) => {
    return [{ name: "webgpu", preferredLayout: layout }];
};

function _setModelInfo(model) {
    if (!model) {
        throw 'Model info is empty';
    }

    if (model.url && model.dataType && model.layout) {
        return model;
    }
    // Only string of model url is provided.
    else if (typeof model == 'string') {
        return {
            url: model,
            dataType: 'float32',
            layout: 'NCHW',
        }
    }
    else {
        throw 'Invalid model param format';
    }
}

// Populates d_in (input data) object.
async function _prepareInput(model, input, width, height) {

    if (input instanceof File) {
        await Image.prepareInputFromFile(input, model);
    }
    else if (input instanceof Uint8Array) {
        await Image.prepareInputFromPixels(input, width, height, model);
    }

    return;
}

// Utilize gpu memory instead cpu ram.
// https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
// Not implemented yet.
function _preAllocWebGPU(scale, dataType) { // both alloc buffer and output tensor
    const outputHeight = (d_in.h * scale);
    const outputWidth = (d_in.w * scale);
    const outputSize = 1 * 3 * outputHeight * outputWidth;
    const bufferSize = outputSize * 4; // 4 bytes per float32 element

    // Access the WebGPU device from onnxruntime-web environment
    const device = _env.webgpu.device;

    // Create a pre-allocated GPU buffer with 16-byte alignment
    const preAllocatedBuffer = device.createBuffer({
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        size: Math.ceil(bufferSize / 16) * 16 // Align to 16 bytes
    });

    // Create the pre-allocated output tensor using the GPU buffer
    const preAllocOutputTensor = ort.Tensor.fromGpuBuffer(preAllocatedBuffer, {
        dataType: dataType,
        dims: [1, 3, outputHeight, outputWidth]
    });

    return preAllocOutputTensor;
}

// Running the session on main thread, return output tensor.
async function _sessionRunner(inputTensor, ModelInfo) {
    const session = await ort.InferenceSession.create(ModelInfo.url, InferenceOpt);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // prepare feeds. use model input tensor names as keys ()
    const feeds = { [inputName]: inputTensor };
    let result;

    // Feed input and run 
    result = await session.run(feeds);

    let outputTensor = result[outputName];
    const imageData = outputTensor.toImageData();
    const tensorBuffer = await outputTensor.getData()

    d_out.imageData = imageData;
    d_out.multiplier = imageData.width / d_in.w;
    d_out.tensor = tensorBuffer;
    d_out.dims = outputTensor.dims;

    outputTensor.dispose(); 
    session.release();
    return;
}

// Running the session on worker, return output image data or buffer. Since only serializable data can be transferred between workers.
async function _sessionRunner_thread(ModelInfo) {
    return new Promise((resolve, reject) => {
        const workerBlob = new Blob([sessionRawWorker], { type: 'application/javascript' });
        const workerURL = URL.createObjectURL(workerBlob);
        const webGPUWorker = new Worker(workerURL);

        webGPUWorker.postMessage({ inputArray: d_in.tensor.buffer, dims: d_in.dims, cfg: cfg, env: _env, ModelInfo: ModelInfo, InferenceOpt: InferenceOpt }, [d_in.tensor.buffer]);

        // Web worker only post result
        webGPUWorker.onmessage = async (event) => {
            if (event.data == 'done') {
                webGPUWorker.postMessage('get-data');  
            }
            if (event.data.tensor) {
                d_out.imageData = event.data.image;
                d_out.tensor = event.data.tensor;
                d_out.multiplier = event.data.image.width / d_in.w
                d_out.dims = event.data.dims;
                webGPUWorker.postMessage('cleanup');
                webGPUWorker.terminate();
                resolve('done');
            }
        };

        webGPUWorker.onerror = (error) => {
            reject(error);
        }
    });
}

// Run the model inference.
// If model (1st param) is only url instead ModelInfo, it will assume the tensor as: NCHW, float32 and 3 channels.
// 2nd param is input, can be uint8array of pixels or File object from file input. For uint8array input, 3rd and 4th param is necessary.
export async function inferenceRun(model, input, inputWidth, inputHeight) {
    try {
        let inputTensor;
        const runtimeIsGPU = InferenceOpt.executionProviders[0] == 'webgpu' && !!navigator.gpu;

        const ModelInfo = _setModelInfo(model);
        await _prepareInput(ModelInfo, input, inputWidth, inputHeight); 

        if (runtimeIsGPU) {
            // Force NCHW for webgpu if input is NCHW, because default is NHWC, this should be on basic onnxweb's docs!
            InferenceOpt.executionProviders = _setWebGpuExecProvider(ModelInfo.layout);
            InferenceOpt.preferredOutputLocation = 'cpu';
        }

        const start = new Date();

        // if prefer to use worker for webgpu/wasm. Let's pray client browser have both.
        if (cfg.wasmGpuRunOnWorker) {
            await _sessionRunner_thread(ModelInfo)
        }
        else {
            inputTensor = new ort.Tensor(ModelInfo.dataType, d_in.tensor, d_in.dims);
            await _sessionRunner(inputTensor, ModelInfo); 
        }

        const end = new Date();
        const inferenceTime = (end.getTime() - start.getTime()) / 1000;
        d_out.time = inferenceTime;
       
        return;
        // improve or rate this workflow :)
    } catch (e) {
        console.error(e);
        throw e;
    }
}

export { Image, meta };

// const dummySession = async() =>{
//     const session = await ort.InferenceSession.create('/model/3/float32/NCHW/4x-ClearRealityV1-fp32-opset14.onnx', InferenceOpt);
//     session.release();
// };

// dummySession();