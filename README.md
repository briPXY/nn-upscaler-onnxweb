# Bundle for inference with ONNX Web Runtime (Image Processing Models)

Even with the most compact models, page often crashed from large input due limited memory, especially tensor with free dimension. This javascript bundle slices an image and run inference on each chunk. Build on top of [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web). Included a web app for demo with image upscaling models.

## Usage

#### Install and build:

```bash
# install dependecies
npm install

# build bundle.min.js (production mode) and copy the backends from node_module
npx webpack

# run the web demo at http://127.0.0.1 (use mainstream browsers for webgpu support)
npm start
``` 
_Select an image -> select models / runtime -> upscale_

Bundle and it's backends will generated in /dist/. Bundle.min.js is lightweight due to it's independent from ort module, the ort modules (recommended to use ort.all.min.js) can be provided externally like from a CDN, by assigning the URL to `wnx.modulePath.all = "path/to/ort.all.min.js"` or set url contain all module/backend file with `wnx.setRuntimePathAll("url/without/filename/)`.

## Runtime Backends

- Web Assembly: Multi-threading/SIMD is supported by onnx runtime.
- WebGPU: Enabled on Chrome on most devices. Mostly faster than wasm especially for dedeicated GPU. If you encounter issues, try `chrome://flags/#enable-unsafe-webgpu` flag or try a browser with official WebGPU support.

Inference session run in the worker by default unless ```wnx.cfg.wasmGpuRunOnWorker = false ```.
There is also webgl runtime but acts unexpected (dims/tensor layout always rejected wether using NCHW or NHWC as input).

## Using the Bundle

### Basic inference

```javascript
// Model instance and required infos.
const myModel = new wnx.Model('https://cdn-domain.com/path/to/ImageUpscaling-2x.onnx');
myModel.dataType: 'float32';
myModel.layout: 'NCHW';
myModel.channel: 3; 

// OutputData instance.
const myOutput = new wnx.OutputData({preserveAlpha: true});

// Start inference with an image file from input.
await wnx.inferenceRun(myModel, fileInput.files[0], myOutput);

// Results.
myOutput.imageData; // contain pixel buffer (Uint8array) and output dimensions
myOutput.tensor;   // raw tensor (TypedArray)
```

### The 'env' Flags and Session Options

Described [here](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#the-env-flags-and-session-options) simply replace 'ort' with 'wnx', since by default the ort module only loaded when running inference. Assigned options will be passed on ort's session.
```javascript 
wnx.env.wasm.proxy = true;
wnx.env.logLevel = "verbose";

// To set session options
wnx.InferenceOpt = {
    executionProviders: ["webgpu", "wasm"]
}
```

In ort module there is option [freeDimensionOverrides](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides) for model with free dimension, this can be set simply with pass dimenstion value to tileSize. This will override default slicing (horizontal). If model required specific dimension then assigning the value is a must.
```javascript
// If freeDimensionOverrides options passed on session option or wnx.InferenceOpt
// Or if model input require specific dimension like [1,3,224,224]
myModel.tileSize = 224;
```

### Configs

Bundles related config.
```javascript
// Force inference on worker, like if env.wasm.proxy is not available from CORS issue
wnx.cfg.wasmGpuRunOnWorker = true;

// Chunk level 1 - 4, lower mean less image data sliced each partial inference on an image
wnx.cfg.avgChunkSize = 2;
```

## Known Issues and Limitations 
- No merged tensor output yet for models with tile based tensors.
- Unmanaged page's memory usage (likely heaps from onnx-runtime) after an inference, equal as youtube page but expected to be more lightweight.
- Inference on browser might less performant than running onnx model natively.

## Included Models

| Architecture| Model| Scale | Tensor | Original Format |
| ----------- | ---------------------- | ----- | -------- | ------- | --------------- |
| SPAN        | [ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)| 4 | float32 | .onnx  |
| Real-ESRGAN | [NomosUni-otf-medium](https://openmodeldb.info/models/2x-NomosUni-compact-otf-medium) | 2 | float32 | .pth |
| SPAN        | [NomosUni-multijpg-ldl](https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl) | 2 | float32 | .pth |

## Additional Informations

### Getting Models and Runtime Informations
- More information about usage of onnxruntime-web (https://onnxruntime.ai/docs/tutorials/web/).
- Building your own onnx runtime for web (https://onnxruntime.ai/docs/build/web.html).
- ONNX javascript API reference (https://onnxruntime.ai/docs/api/js/index.html).
- Guide to converts models into ONNX format (https://onnx.ai/onnx/intro/converters.html).

Putting your own model in model subfolders ordered by formats make your model automatically appear in the selection. To acquire upscaling model informations, if you're not developer/trainer use model viewer app/vscode-plugin like [netron.app](https://netron.app).
 
### Example Application
[https://pixaya.app](https://pixaya.app)

### Models & 3rd Party Credits
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- [fastify/fastify](https://github.com/fastify/fastify)