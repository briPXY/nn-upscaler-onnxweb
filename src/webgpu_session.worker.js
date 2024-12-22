//
// A worker for a webgpu/wasm session running.
// 

let outputName, result, session, inputTensor, outputTensor;

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
            return
        }

        const data = event.data;
        const provider = data.InferenceOpt.executionProviders[0] == 'wasm' ? 'wasm' : 'webgpu';
        const array32 = new Float32Array(data.inputArray);

        // import budle from /dist/  
        importScripts(data.cfg.backendPath[provider]);

        setEnv(data.env);

        session = await ort.InferenceSession.create(data.ModelInfo.url, data.InferenceOpt);
        inputTensor = new ort.Tensor(data.ModelInfo.dataType, array32, data.dims);
        const inputName = session.inputNames[0];
        outputName = session.outputNames[0];

        // prepare feeds. use model input tensor names as keys ()
        const feeds = { [inputName]: inputTensor };

        // Feed input and run 
        result = await session.run(feeds);
        self.postMessage('done');
        return;
    }
    catch (e) {
        console.error(e);
        throw e;
    }
})

self.onerror = (message, source, lineno, colno, error) => {
    console.error("Worker Error:", { message, source, lineno, colno, error });
    // Handle or propagate the error as needed
};

self.onunhandledrejection = (event) => { 
    if (event.reason && event.reason.message.includes("GPUBuffer")) {
        console.error("WebGPU Error in Worker");
        self.postMessage({gpuError: event.reason.message})
    }
};