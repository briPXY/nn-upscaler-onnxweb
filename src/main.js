// import * as ort from 'onnxruntime-web';  // Bundling not includes webgpu backend despite stated true.
import * as Image from './image_helper';
import * as meta from './meta';
import { cfg } from './cfg';
import { OutputData, Dims, ChunkLevel, Model } from './types';

// worker raw
import sessionRawWorker from './webgpu_session.worker';

// To be set on ort.env.
export const _env = {
    wasm: {},
};

_env.logLevel = 'info';
_env.wasm.wasmPaths = `${cfg._defaultModulePath}`;
_env.wasm.proxy = false;
_env.wasm.numThreads = meta.threads.default;

export const InferenceOpt = {
    executionProviders: meta.providers,
};

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

const _setWebGpuExecProvider = (layout) => {
    return [{ name: "webgpu", preferredLayout: layout }];
};

function _validateParam(model, output) {
    if (!model instanceof Model) {
        throw '1st param must be an instance of NNU.Model';
    }

    if (!output instanceof OutputData) {
        throw '3rd param must be an instance of NNU.OutputData';
    }

    model.validate();
}

// Populates d_in (input data) with chunks of tensor.
async function _prepareInput(model, input, width, height) {

    if (input instanceof File) {
        handlers.chunkProcess.total = await Image.prepareInputFromFile(input, model);
        return;
    }

    else if (input instanceof Uint8Array && width && height) {
        if (!width || !height) {
            throw 'Width amd height are required for TypedArray input (Uint8Array).'
        }
        await Image.prepareInputFromPixels(input, width, height, model);
        return;
    }

    else if (input.tensor) {
        if (input.dataType && input.layout && input.channels) {
            const typeMatch = input.dataType == model.dataType;
            const layoutMatch = input.layout == model.layout;
            const channelMatch = input.channels == model.channel;

            if (typeMatch && layoutMatch && channelMatch) {
                Image.prepareInputFromTensor(input, width, height, model);
                return;
            }
            else {
                throw `Tensor formats does not match with model. \n Model: type-${model.dataType}, layout-${model.layout}, channel-${model.channel} \n Input: type-${input.dataType}, layout-${input.layout}, channel-${input.channels}`;
            }
        }
        else {
            throw 'Invalid input object props for tensor input';
        }
    }
    throw 'Invalid input params';
}


// Running the session on main thread, return output tensor.
async function _sessionRunner(ModelInfo, output) {
    const input = d_in.shift();
    const inputTensor = new ort.Tensor(ModelInfo.dataType, input.tensor, input.dims);
    const session = await ort.InferenceSession.create(ModelInfo.url, InferenceOpt);

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // prepare feeds. use model input tensor names as keys ()
    const feeds = { [inputName]: inputTensor };

    // Feed input and run 

    //console.log('process chunk no:', d_in.length, input.dims, 'height:', input.h);
    const result = await session.run(feeds);
    handlers.chunkProcess.doneEvent();

    let outputTensor = result[outputName];
    const imageData = outputTensor.toImageData();
    const tensorBuffer = await outputTensor.getData()
    output.tensor = Image.mergeTensors[ModelInfo.layout](output.tensor, tensorBuffer, output.imageData.height, imageData.height, imageData.width, ModelInfo.channel, ModelInfo.dataType);

    output.imageData.data = Image.concatUint8Arrays(output.imageData.data, imageData.data);
    output.imageData.width = imageData.width;
    output.imageData.height += imageData.height;

    inputTensor.dispose();
    outputTensor.dispose();
    session.release();

    if (d_in.length > 0) {
        await _sessionRunner(ModelInfo);
    }

    output.dims = Dims(ModelInfo.layout, { W: output.imageData.width, H: output.imageData.height, C: ModelInfo.channel, N: 1 });
    output.multiplier = output.imageData.width / input.w;

    return;
}

// Running the session on worker, return output image data or buffer. Since only serializable data can be transferred between workers.
async function _sessionRunner_thread(ModelInfo, output) {
    return new Promise((resolve, reject) => {
        const workerBlob = new Blob([sessionRawWorker], { type: 'application/javascript' });
        const workerURL = URL.createObjectURL(workerBlob);
        const webGPUWorker = new Worker(workerURL);

        webGPUWorker.postMessage({
            inputArray: d_in[0].tensor.buffer,
            dims: d_in[0].dims,
            cfg: cfg,
            env: _env,
            ModelInfo: ModelInfo,
            InferenceOpt: InferenceOpt
        }, [d_in[0].tensor.buffer]);

        // Web worker only post result
        webGPUWorker.onmessage = async (event) => {
            if (event.data == 'chunk done') {
                handlers.chunkProcess.doneEvent();
                webGPUWorker.postMessage('get-data');
            }

            if (event.data.tensor) {
                d_in.shift();
                const imageData = event.data.image;
                output.tensor = Image.mergeTensors[ModelInfo.layout](output.tensor, event.data.tensor, output.imageData.height, imageData.height, imageData.width, ModelInfo.channel, ModelInfo.dataType);
                output.imageData.data = Image.concatUint8Arrays(output.imageData.data, imageData.data);
                output.imageData.width = imageData.width;
                output.imageData.height += imageData.height;

                if (d_in.length === 0) {
                    output.multiplier = imageData.width / d_in.w
                    output.dims = Dims(ModelInfo.layout, { W: output.imageData.width, H: output.imageData.height, C: ModelInfo.channel, N: 1 });
                    webGPUWorker.postMessage('cleanup');
                    webGPUWorker.terminate();
                    resolve('done');
                    return;
                }

                // Cycle for next chunk.
                webGPUWorker.postMessage('cleanup');
                webGPUWorker.postMessage({
                    inputArray: d_in[0].tensor.buffer,
                    dims: d_in[0].dims,
                }, [d_in[0].tensor.buffer]);

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
 * @param {Object} input; - Can be uint8array of pixels or File object from file input. For uint8array input, 3rd and 4th param is necessary.  
 * @param {number} inputWidth - Only required if using TypedArray as input.
 * @param {number} inputHeight - Only required if using TypedArray as input.
 */
export async function inferenceRun(model, input, output, inputWidth, inputHeight) {
    try {
        const runtimeIsGPU = InferenceOpt.executionProviders[0] == 'webgpu' && !!navigator.gpu;

        _validateParam(model, output);
        await _prepareInput(model, input, inputWidth, inputHeight);
        _setEnv();

        if (runtimeIsGPU) {
            // Force NCHW for webgpu if input is NCHW, because default is NHWC, this should be on basic onnxweb's docs!
            ort.env.wasm.proxy = false;
            InferenceOpt.executionProviders = _setWebGpuExecProvider(model.layout);
            InferenceOpt.preferredOutputLocation = 'cpu';
        }

        if (!cfg.wasmGpuRunOnWorker) {
            await meta.loadBackendScript();
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

export { Image, meta, Model, OutputData, ChunkLevel, cfg };