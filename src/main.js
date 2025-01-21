import * as Image from './image_helper';
import * as meta from './meta';
import { cfg, modulePath, _env } from './cfg';
import { OutputData, Dims, ChunkLevel, Model, TypedArray } from './types';

// worker raw
import sessionRawWorker from './session.worker';

export const env = _env;

export const InferenceOpt = {
    executionProviders: meta.providers,
};

// Set pre-inference defined ort.env values.
function _setEnv() {
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
}

// Input data/tensor chunks.
export const d_in = [];

// Must be a funciton
export const handlers = {
    gpuInferenceError: () => { },
    chunkProcess: {
        total: 0,
        doneEvent: () => { },
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

export function setEnvFlag(props) {
    for (const prop in props) {
        _env[prop] = props[prop];
    }
}

export function setInferenceOption(obj = {}) {
    for (const prop in obj) {
        InferenceOpt[prop] = obj[prop];
    }
    return;
}

// Set path for onnxweb runtime distributions. Will override all runtime provider path.
export function setRuntimePathAll(url) {
    /\/$/.test(url) ? url : url += '/';
    cfg._defaultModulePath = url;
    modulePath.all = `${url}ort.all.min.js`;
    modulePath.webgpu = `${url}ort.webgpu.min.js`;
    modulePath.wasm = `${url}ort.wasm.min.js`;
}

const _setWebGpuExecProvider = (layout) => {
    return [{ name: "webgpu", preferredLayout: layout }];
};

function _validateParam(model, output) {
    if (!model instanceof Model) {
        throw '1st param must be an instance of wnx.Model';
    }

    if (!output instanceof OutputData) {
        throw '3rd param must be an instance of wnx.OutputData';
    }

    model.validate();
}

// Populates d_in with chunks of sliced pixels.
// Assign pre-result OutputData's with required data for post-result operation.
async function _prepareInputOutput(model, input, output, width, height) {
    let result;

    if (input instanceof File) {
        result = await Image.prepareInputFromFile(input, model);
    }

    else if (input instanceof Uint8Array || input instanceof Uint8ClampedArray) {
        if (!width || !height) {
            throw 'Invalid input params -- width and height are required for TypedArray input (Uint8Array).'
        }
        result = await Image.prepareInputFromPixels(input, width, height, model);
    }
    else {
        throw "Input error -- input type is not supported";
    }

    handlers.chunkProcess.total = result.totalChunks;
    output.prevData = result.prevData;
    output.insert = result.insert;
    output._prePadding = result.prePadding;
    output._tileDim = result.tileDim;

    return;
}

async function _createTensor(input, model) {
    let bitmap = await Image.createImageBitmapFromRGB(input.w, input.h, new Uint8Array(input.data));
    const tensor = await ort.Tensor.fromImage(bitmap, {
        dataType: model.dataType,
        tensorLayout: model.layout.toUpperCase(),
        tensorFormat: model.channel == 3 ? 'RGB' : 'RGBA',
    });
    bitmap = null;
    input = null;
    return tensor;
}

// Running the session on main thread, return output tensor.
async function _sessionRunner(ModelInfo, output) {
    const input = d_in.shift();

    const inputTensor = await _createTensor(input, ModelInfo);

    const session = await ort.InferenceSession.create(ModelInfo.url, InferenceOpt);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // prepare feeds. use model input tensor names as keys ()
    const feeds = { [inputName]: inputTensor };

    const result = await session.run(feeds);
    handlers.chunkProcess.doneEvent();

    let outputTensor = result[outputName];
    const imageData = outputTensor.toImageData();
    const tensorBuffer = await outputTensor.getData();

    output.insertTensorChunk(tensorBuffer, null, imageData.height, ModelInfo);
    output.insertImageChunk(imageData);

    inputTensor.dispose();
    outputTensor.dispose();
    session.release();

    if (d_in.length > 0) {
        await _sessionRunner(ModelInfo, output);
    }

    output.finish();

    return;
}

// Running the session on worker, return output image data or buffer. Since only serializable data can be transferred between workers.
async function _sessionRunner_thread(ModelInfo, output) {
    return new Promise(async (resolve, reject) => {
        const workerBlob = new Blob([sessionRawWorker], { type: 'application/javascript' });
        const workerURL = URL.createObjectURL(workerBlob);
        const webGPUWorker = new Worker(workerURL);
        const bitmap = await Image.createImageBitmapFromRGB(d_in[0].w, d_in[0].h, new Uint8Array(d_in[0].data));

        webGPUWorker.postMessage({
            bitmap: bitmap,
            dims: d_in[0].dims,
            cfg: cfg,
            env: _env,
            modulePath: InferenceOpt.executionProviders[0] == 'wasm' ? modulePath._wasm : modulePath.webgpu,
            ModelInfo: ModelInfo,
            InferenceOpt: InferenceOpt
        }, [bitmap]);

        // Web worker only post result
        webGPUWorker.onmessage = async (event) => {
            if (event.data == 'chunk done') {
                handlers.chunkProcess.doneEvent();
                webGPUWorker.postMessage('get-data');
            }

            if (event.data.tensor) {
                d_in.shift();
                const imageData = event.data.image;
                output.insertTensorChunk(event.data.tensor, null, imageData.height, ModelInfo);
                output.insertImageChunk(imageData);

                if (d_in.length === 0) {
                    output.finish();
                    webGPUWorker.postMessage('cleanup');
                    webGPUWorker.terminate();
                    resolve('done');
                    return;
                }

                // Cycle next chunk.
                const bitmap = await Image.createImageBitmapFromRGB(d_in[0].w, d_in[0].h, new Uint8Array(d_in[0].data));
                webGPUWorker.postMessage('cleanup');
                webGPUWorker.postMessage({
                    bitmap: bitmap,
                    dims: d_in[0].dims,
                }, [bitmap]);

            }
            if (event.data.gpuError) {
                handlers.gpuInferenceError();
                reject({ gpuError: event.data.gpuError });
            }
        };

        webGPUWorker.onerror = (error) => {
            reject(error);
        }
    });
}


/**
 * Run the model inference.
 * @param {Object} model - ONNX model information from Model instance.
 * @param {Object} input; - Can be uint8array of pixels or File object from file input. For typed array input, 3rd and 4th param is necessary.  
 * @param {number} inputWidth - Only required if using TypedArray as input.
 * @param {number} inputHeight - Only required if using TypedArray as input.
 */
export async function inferenceRun(model, input, output, inputWidth, inputHeight) {
    try {
        const runtimeIsGPU = InferenceOpt.executionProviders[0] == 'webgpu' && !!navigator.gpu;

        _validateParam(model, output);
        await _prepareInputOutput(model, input, output, inputWidth, inputHeight);
        output.tensor = TypedArray[model.dataType](0);
        output.model = model;

        if (runtimeIsGPU) {
            // Force NCHW for webgpu if input is NCHW, because default is NHWC, this should be on basic onnxweb's docs!
            setWasmFlags({ proxy: false });
            InferenceOpt.executionProviders = _setWebGpuExecProvider(model.layout);
            InferenceOpt.preferredOutputLocation = 'cpu';
        }

        if (!cfg.wasmGpuRunOnWorker || !window.ort) {
            await meta.loadBackendScript();
            _setEnv();
        }

        const start = new Date();

        // if prefer to use worker for webgpu/wasm. Let's pray client browser have both.
        if (cfg.wasmGpuRunOnWorker) {
            await _sessionRunner_thread(model, output)
        }
        else {
            await _sessionRunner(model, output);
        }

        const end = new Date();
        const inferenceTime = (end.getTime() - start.getTime()) / 1000;
        output.time = inferenceTime;

        return;
        // improve or rate this workflow :)
    } catch (e) {
        console.error(e);
        throw e;
    }
}

export { Image, meta, Model, OutputData, ChunkLevel, cfg, modulePath };

Object.defineProperty(window, 'NNU', {
    get: function () {
        console.warn("The module name 'NNU' has been changed to 'wnx'. Please use 'wnx' instead.");
        return wnx;
    },
    configurable: true
});