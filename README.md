## Neural Networks on Browser - ONNX Web Runtime Demo

Image upscaling with neural network models, in the browser, using onnxruntime-web.

### Usage

#### Install and build:
```bash
# install dependecies
npm install
# build the bundle and copy runtime files
npx webpack 
```
#### Run the web server:
```bash
npm start
# open http://127.0.0.1 (use latest mainstream browser for webgpu support)
```
Select an image -> select models / runtime -> upscale

### Runtime Backends
- Web Assembly: Multi-threading/SIMD is supported by onnx runtime.
- WebGPU: Enabled on Chrome on most devices. Mostly faster than wasm. If you encounter issues, try `chrome://flags/#enable-unsafe-webgpu` flag or try a browser with official WebGPU support.
Inference session run in the worker by default unless ```NNU.cfg.wasmGpuRunOnWorker = false ```. There is also webgl runtime but act unexpected (dims/tensor layout always rejected wether using NCHW or NHWC as input).


### Using the Bundle
#### Basic inference
```javascript
// create model info object, 4 props below are required to be passed
const ModelInfo = {
    url: 'https://cdn-domain.com/path/to/model.onnx',
    dataType: 'float32',
    layout: 'NCHW', 
    channel: 3,   // number, refer to input dims
};

// start inference, with a file object (image file)
await NNU.inferenceRun(ModelInfo, fileInputDom.files[0]);

// the output data are stored in:
NNU.d_out.imageData // contain pixel buffer (Uint8array) and output dimensions
NNU.d_out.tensor    // raw tensor in float (eg: Float32array)
```

#### Pre-inference settings
Set wasm [flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), example:
```javascript
NNU.setWasmFlags({wasmPaths: 'cdn/of/onnxruntime-web/dist/', numThreads: 6})
```
Set session [options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), example:
```javascript
NNU.setInferenceOption({executionProviders: ['wasm']})
```

#### Post-inference utility
Encodes image data from output, return a blob, example
```javascript
const pixels = NNU.d_out.imageData.data;
const width = NNU.d_out.imageData.width;
// etc
const blob = await NNU.Image.encodeRGBA(pixels, width, height, 90, 'webp')
```
Convert output tensor (nchw) to image data (rgb), return an Uint16Array.
```javascript
const imageData16 = await NNU.Image.tensorToRGB16_NCHW(NNU.d_out.tensor)
```

### Included Models

| Architecture| Model                  | Scale | Tensor   | Layout  | Original Format |
| ----------- | ---------------------- | ----- | -------- | ------- | --------------- |
| SPAN        | ClearRealityV1         | 4     | float32  | NCHW    | .onnx           |
| Real-ESRGAN | NomosUni-otf-medium    | 2     | float32  | NCHW    | .pth            |
| SPAN        | NomosUni-multijpg-ldl  | 2     | float32  | NCHW    | .pth            |


### Additional Informations

- More information about usage of onnxruntime-web (https://onnxruntime.ai/docs/tutorials/web/).
- Building your own onnx runtime for web (https://onnxruntime.ai/docs/build/web.html).
- ONNX javascript API reference (https://onnxruntime.ai/docs/api/js/index.html).
- Guide to converts models into ONNX format (https://onnx.ai/onnx/intro/converters.html).
Informations about the models can be acquired on developer/trainer pages or using tool/vscode plugin like [netron.app](https://netron.app).
 
### Example Application
[https://pixaya.app](https://pixaya.app)

### Models & 3rd Party Credits

- [ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1) / [SPAN](https://github.com/hongyuanyu/SPAN)
- [NomosUni compact otf medium](https://openmodeldb.info/models/2x-NomosUni-compact-otf-medium)
- [NomosUni span multijpg ldl](https://openmodeldb.info/models/2x-NomosUni-span-multijpg-ldl) / [SPAN](https://github.com/hongyuanyu/SPAN)
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- [fastify/fastify](https://github.com/fastify/fastify)