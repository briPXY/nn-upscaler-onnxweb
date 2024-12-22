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
    threads.optimal = navigator.hardwareConcurrency - 2;
    threads.min = 2;
    threads.default = Math.min(navigator.hardwareConcurrency - 2, 6);
}

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