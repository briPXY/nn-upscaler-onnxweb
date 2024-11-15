// Client infos & utilities

export const providers = [];
export const threads = {};
export const _defaultModulePath = document.currentScript.src.match(/.+\//)[0];
export const _domain = document.currentScript.src.match(/(http|https):\/\/[^/]+/)[0];

if (navigator.gpu) {
    providers.push('webgpu');
}

if (typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function") {
    providers.push('wasm');
}

// Only fetch headers.
export async function _checkUrlExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        if (response.ok) {
            return true;
        }
        else {
            return false;
        }
    } catch (error) {
        throw 'Model url not exist';
    }
}

if (navigator.hardwareConcurrency) {
    threads.max = navigator.hardwareConcurrency - 2;
    threads.min = 2;
    threads.default = Math.min(navigator.hardwareConcurrency - 2, 4);
}