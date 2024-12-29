// Client infos & utilities
import { cfg } from "./cfg";

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

		script.src = cfg.backendPath[provider];
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

//#region Benchmarks

// Benchmarks maybe useful for progress info.
export async function GPUBenchmark() {
	if (!navigator.gpu) {
		throw new Error("WebGPU not supported in this browser.");
	}

	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// Shader code to compute square roots
	const shaderCode = `
    @group(0) @binding(0) var<storage, read> input : array<f32>;
    @group(0) @binding(1) var<storage, write> output : array<f32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
      let index = global_id.x;
      if (index < arrayLength(&input)) {
        output[index] = sqrt(input[index]);
      }
    }
  `;

	// Compile shader
	const shaderModule = device.createShaderModule({ code: shaderCode });

	// Define data
	const iterations = 10_000_000; // Matches CPU workload
	const inputData = new Float32Array(iterations).map((_, i) => i);
	const outputData = new Float32Array(iterations);

	// Create buffers
	const inputBuffer = device.createBuffer({
		size: inputData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	const outputBuffer = device.createBuffer({
		size: outputData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	// Upload input data
	device.queue.writeBuffer(inputBuffer, 0, inputData);

	// Create bind group
	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
			{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
		],
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	// Create compute pipeline
	const pipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
		compute: { module: shaderModule, entryPoint: "main" },
	});

	// Create command encoder and dispatch
	const commandEncoder = device.createCommandEncoder();
	const computePass = commandEncoder.beginComputePass();
	computePass.setPipeline(pipeline);
	computePass.setBindGroup(0, bindGroup);
	computePass.dispatchWorkgroups(Math.ceil(iterations / 256));
	computePass.end();

	const start = performance.now();

	// Submit and wait for GPU to finish
	device.queue.submit([commandEncoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	const end = performance.now();

	// Download output data (optional, to verify computation)
	const readBuffer = device.createBuffer({
		size: outputData.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputData.byteLength);
	device.queue.submit([commandEncoder.finish()]);


	// Output results
	const duration = (end - start).toFixed(2);
	const throughput = (iterations / (duration / 1000)).toFixed(2); // ops/sec

	return { time: duration, counts: throughput }; // Output first 10 results for verification
}



export async function CPUBenchmark() {

	const iterations = 10_000_000; // Smaller workload
	const start = performance.now();

	// Perform a simple computation
	let result = 0;
	for (let i = 0; i < iterations; i++) {
		result += i ** 0.5; // Square root computation
	}

	const end = performance.now();

	// Output results
	const duration = (end - start).toFixed(2);
	const throughput = (iterations / (duration / 1000)).toFixed(2); // ops/sec
	return { time: duration, counts: throughput };
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