//
// A worker for a webgpu/wasm session running.
// 

let outputName, result, session, inputTensor, outputTensor;
let ModelInfo, InferenceOpt, env, cfg;

function setEnv(env) {
    for (const key in env) {
        if (typeof env[key] !== 'object') {
            ort.env[key] = env[key];
        }
    }

    for (const key in env.wasm) {
        ort.env.wasm[key] = env.wasm[key];
    }

    if (env.webgpu) {
        for (const key in env.webgpu) {
            ort.env.webgpu[key] = env.webgpu[key];
        }
    }
}

self.addEventListener('message', async (event) => {
    try {
        if (event.data == 'cleanup') {
            // cleanup
            session.release();
            inputTensor.dispose();
            outputTensor.dispose();
            return;
        }

        if (event.data == 'get-data') {
            outputTensor = result[outputName];
            const tensorBuffer = await outputTensor.getData();
            const imageData = outputTensor.toImageData();
            self.postMessage({ image: imageData, tensor: tensorBuffer, dims: outputTensor.dims });
            return;
        }

        event.data.InferenceOpt ? InferenceOpt = event.data.InferenceOpt : InferenceOpt;
        event.data.ModelInfo ? ModelInfo = event.data.ModelInfo : ModelInfo;
        event.data.env ? env = event.data.env : env;
        event.data.cfg ? cfg = event.data.cfg : cfg; 

        // Import backend provider script  
        if (typeof ort == 'undefined') {
            importScripts(event.data.modulePath);
        }

        setEnv(env);

        session = await ort.InferenceSession.create(ModelInfo.url, InferenceOpt);
        inputTensor = await ort.Tensor.fromImage(event.data.bitmap, { 
            dataType: ModelInfo.dataType, 
            tensorLayout: ModelInfo.layout.toUpperCase(), 
            tensorFormat: ModelInfo.channel == 3? 'RGB' : 'RGBA',
        });
        const inputName = session.inputNames[0];
        outputName = session.outputNames[0];

        // prepare feeds. use model input tensor names as keys ()
        const feeds = { [inputName]: inputTensor };

        // Feed input and run 
        result = await session.run(feeds);
        self.postMessage('chunk done');
        return;
    }
    catch (e) {
        console.error(e);
        throw e;
    }
});

self.onerror = (message, source, lineno, colno, error) => {
    console.error("Worker Error:", { message, source, lineno, colno, error });
    // Handle or propagate the error as needed
};

self.onunhandledrejection = (event) => {
    if (event.reason && event.reason.message.includes("GPUBuffer")) {
        console.error("WebGPU Error in Worker");
        self.postMessage({ gpuError: event.reason.message });
    }
};