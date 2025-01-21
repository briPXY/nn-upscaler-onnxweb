// Client infos & utilities
import { modulePath } from "./cfg";

export const providers = [];
export const threads = {};
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

//#region Load script for backends.

export async function loadBackendScript(provider = 'all') {
	return new Promise((resolve, reject) => {
		let script = document.getElementById('onnx-backend');

		if (!script) {
			script = document.createElement('script');
			script.id = 'onnx-backend';
		}

		if (script.src != '') {
			resolve('loaded');
			return;
		}

		script.src = modulePath[provider];
		script.onload = () => {
			resolve('loaded');
		};
		script.onerror = (error) => {
			console.error(`loadRuntimeBackend() error @ loading ${provider} backend`);
			reject(error);
		};
	});
}

//#endregion

if (navigator.hardwareConcurrency) {
	threads.optimal = navigator.hardwareConcurrency - 2;
	threads.min = 2;
	threads.default = Math.min(navigator.hardwareConcurrency - 2, 6);
}

// future implementation for IndexedDB model storing
const _ModelCollections = {};

// This server use only. Return models info from all model stored in static folder in ordered directory (/channel/datatype/dims/modelfile.onnx).
// Useful for automating list element.
export async function requestModelsCollections(api_url) {
	try {
		const response = await fetch(api_url);

		if (!response.ok) {
			throw new Error(`HTTP error! Status: ${response.status}`);
		}
		const modelPaths = await response.json();

		modelPaths.forEach(url => {
			const pathText = url.match(/[a-zA-Z0-9\.\_\-]+/g);
			const l = pathText.length;
			const name = pathText[l - 1].replace('.onnx', '');
			_ModelCollections[name] = {
				channel: parseInt(pathText[l - 4]),
				dataType: pathText[l - 3],
				layout: pathText[l - 2].toUpperCase(),// tensor format
				url: _domain + url.replace(/static|\\/g, '/').replace('//', '/'),
				rawPath: url,
			};
		});

		return _ModelCollections;
	} catch (error) {
		console.error('Error fetching filenames:', error);
	}
}


// Get model info into ModelInfo. Using formatted URL. 
// Path should in an format of '.../channel/dataType/tensorFormat/model-name.onnx', URL must ended with .onnx.
export async function makeModelInfo_formatted(modelUrl) {
	try {
		const response = await _checkUrlExists(modelUrl);

		if (!response) {
			throw new Error(`Model URL ${modelUrl} not exit!`);
		}

		const ModelInfo = {}
		const pathText = modelUrl.match(/([^\/]+)/g);

		if (!(/http|https/).test(pathText[0])) {
			throw 'Invalid model url path, use absolute path';
		}

		const l = pathText.length;
		const name = pathText[l - 1].replace('.onnx', '');
		ModelInfo.name = name;
		ModelInfo.channel = parseInt(pathText[l - 4]);
		ModelInfo.dataType = pathText[l - 3];
		ModelInfo.layout = pathText[l - 2].toUpperCase();
		ModelInfo.url = modelUrl;

		return ModelInfo;
	} catch (error) {
		console.error('Error fetching url:', error);
		throw error;
	}
}

// Utilize gpu memory instead cpu ram.
// https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
// Not implemented yet.
export async function _preAllocWebGPU(scale, dataType, d_in) { // both alloc buffer and output tensor
	const outputHeight = (d_in.h * scale);
	const outputWidth = (d_in.w * scale);
	const outputSize = 1 * 3 * outputHeight * outputWidth;
	const bufferSize = outputSize * 4; // 4 bytes per float32 element

	// Access the WebGPU device from onnxruntime-web environment
	const device = ort.env.webgpu.device;

	// Create a pre-allocated GPU buffer with 16-byte alignment
	const preAllocatedBuffer = device.createBuffer({
		usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
		size: Math.ceil(bufferSize / 16) * 16 // Align to 16 bytes
	});

	// Create the pre-allocated output tensor using the GPU buffer
	const preAllocOutputTensor = ort.Tensor.fromGpuBuffer(preAllocatedBuffer, {
		dataType: dataType,
		dims: [1, 3, outputHeight, outputWidth]
	});

	return preAllocOutputTensor;
}