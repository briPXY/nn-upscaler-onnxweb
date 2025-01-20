# Web Neural Inference for Image Processing Models (ONNX)

Easy interface for running onnx models inference on the web browser, for image processing models. Using ONNX web runtime API.
Features: Inference on each sliced images for AI models which input tensor's dimension is non-determined, to prevent page crash from large input/model (browser's memory limit policy).

## Usage

#### Install and build:
```bash
# install dependecies
npm install

# build bundle.min.js (production mode) and copy the backends from node_module
npx webpack

# run the fastify server for demo at http://127.0.0.1 (use mainstream browsers for webgpu support)
npm start
``` 
_Select an image -> select models / runtime -> upscale_

Bundle and it's backends will generated in static/, bundle.min.js is lightweight due to it's independent from ort module, the javascript backends and it's wasm URL can be provided externally like from a CDN, by assigning the urls on `wnx.backendPath.wasm` or `wnx.backendPath.ort` or `wnx.backendPath.all`.

## Runtime Backends
- Web Assembly: Multi-threading/SIMD is supported by onnx runtime.
- WebGPU: Enabled on Chrome on most devices. Mostly faster than wasm. If you encounter issues, try `chrome://flags/#enable-unsafe-webgpu` flag or try a browser with official WebGPU support.

Inference session run in the worker by default unless ```wnx.cfg.wasmGpuRunOnWorker = false ```.
There is also webgl runtime but acts unexpected (dims/tensor layout always rejected wether using NCHW or NHWC as input).

## Using the Bundle
#### Basic inference
```javascript
// Model instance and required infos.
const myModel = new wnx.Model('https://cdn-domain.com/path/to/model.onnx');
myModel.dataType: 'float32';
myModel.layout: 'NCHW';
myModel.channel: 3;

// If the model is tile based (like Real-ESRGAN-General), assigning tileSize is required.
myModel.tileSize = 128;

// OutputData instance.
const output = new wnx.OutputData({preserveAlpha: true});

// Start inference with an image file from input.
await wnx.inferenceRun(myModel, fileInput.files[0], output);

// Get result.
output.imageData; // contain pixel buffer (Uint8array) and output dimensions
output.tensor;   // raw tensor (TypedArray)
```

#### Pre-inference settings
Set wasm [flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), example:
```javascript
wnx.setWasmFlags({wasmPaths: 'cdn/of/onnxruntime-web/dist/', numThreads: 6})
```
Set session [options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), example:
```javascript
wnx.setInferenceOption({executionProviders: ['wasm']})
```

#### Post-inference utility
Encodes image data from output, return a blob, example
```javascript
const pixels = output.imageData.data;
const width = output.imageData.width;
// etc
const blob = await wnx.Image.encodeRGBA(pixels, width, height, 90, 'webp')
```
Convert output tensor (nchw) to image data (rgb), return an Uint16Array.
```javascript
const imageData16 = await wnx.Image.tensorToRGB16_NCHW(output.tensor)
```

## Known Issues and Limitations 
- No merged tensor output yet for models with tile based tensors.
- Unmanaged page's memory usage (likely heaps from onnx-runtime) after an inference, equal as youtube page but expected to be more lightweight.
- Inference on browser might less performant than running onnx model natively.

## Included Models

| Architecture| Model| Scale | Tensor   | Layout  | Original Format |
| ----------- | ---------------------- | ----- | -------- | ------- | --------------- |
| SPAN        | [ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1)| 4     | float32  | NCHW    | .onnx  |
| Real-ESRGAN | [NomosUni-otf-medium](https://openmodeldb.info/models/2x-NomosUni-compact-otf-medium)    | 2     | float32  | NCHW    | .pth   |
| SPAN        | [NomosUni-multijpg-ldl](https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl)  | 2     | float32  | NCHW    | .pth   |


## Additional Informations


### Additional Models and Runtime Informations
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