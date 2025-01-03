import { ChunkLevel } from "./types";
import { threads } from "./meta";

export const cfg = {
    webGpuBufferLocation: 'cpu',
    wasmGpuRunOnWorker: true, // true = run inference session on worker (not available for webgl)
    _avgChunkSize: ChunkLevel[2], // Average chunk size in pixels, for example 1600p will be sliced for each 400x400px, 0 = one time inference
    _defaultModulePath: document.currentScript.src.match(/.+\//)[0],
};

export const backendPath = {
    webgpu: `${cfg._defaultModulePath}ort.webgpu.min.js`,
    _wasm: `${cfg._defaultModulePath}ort.wasm.min.js`,
    all: `${cfg._defaultModulePath}ort.all.min.js`,
};

// To be set on ort.env.
export const _env = {
    wasm: {},
};

// Set starter ort.env values.
_env.logLevel = 'info';
_env.wasm.wasmPaths = `${cfg._defaultModulePath}`;
_env.wasm.proxy = false;
_env.wasm.numThreads = threads.default;

Object.defineProperty(backendPath, 'wasm', {
    get: function () {
        return this._wasm;
    },
    set: function (path) {
        try {
            this._wasm = path;
            _env.wasm.wasmPaths = path.match(/.+\//)[0];
        } catch (e) {
            console.error('Assign a proper URL')
        }
    }
});

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
