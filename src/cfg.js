import { ChunkLevel } from "./types";

export const _defaultModulePath = document.currentScript.src.match(/.+\//)[0];

export const cfg = {
    webGpuBufferLocation: 'cpu',
    wasmGpuRunOnWorker: true, // true = run inference session on worker (not available for webgl)
    backendPath: {
        webgpu: `${_defaultModulePath}ort.webgpu.min.js`,
        wasm: `${_defaultModulePath}ort.wasm.min.js`,
        all: `${_defaultModulePath}ort.all.min.js`
    },
    _avgChunkSize: ChunkLevel[2], // Average chunk size in pixels, for example 1600p will be sliced for each 400x400px, 0 = one time inference
};

Object.defineProperty(cfg, 'avgChunkSize', {
    get: function () {
        return this._avgChunkSize;
    },
    set: function (size) {
        if (ChunkLevel[size]) {
            this._avgChunkSize = ChunkLevel[size];
        } else {
            this._avgChunkSize = ChunkLevel[1];
            console.warn('Value must be a number between 1 - 4');
        }
    }
});
