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
        console.log("WebGPU not supported on this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    // Create a simple compute shader
    const shaderCode = `
      @group(0) @binding(0) var<storage, read_write> data : array<f32>;
  
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id : vec3<u32>) {
        let index = id.x;
        if (index < arrayLength(&data)) {
          data[index] = data[index] + 1.0;
        }
      }
    `;

    // Create a GPU buffer
    const elementCount = 1024 * 1024; // 1M elements
    const buffer = device.createBuffer({
        size: elementCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    const arrayBuffer = new Float32Array(buffer.getMappedRange());
    arrayBuffer.fill(0);
    buffer.unmap();

    // Create a pipeline
    const module = device.createShaderModule({ code: shaderCode });
    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
    });

    // Create a bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
    });

    // Record commands
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(elementCount / 64)); // Match the workgroup size
    passEncoder.end();

    // Measure execution time
    const start = performance.now();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone(); // Wait for GPU to finish
    const end = performance.now();

    console.log(`WebGPU Benchmark: ${(end - start).toFixed(2)} ms for ${elementCount} elements.`);
    return { time: end - start, counts: elementCount };
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