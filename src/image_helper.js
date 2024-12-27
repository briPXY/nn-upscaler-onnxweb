import { d_in, cfg } from './main'
import workerCodeRawString from './image_helper.worker.js';
import { TypedArray } from './types.js';

// Create the Blob and Worker
const workerBlob = new Blob([workerCodeRawString], { type: 'application/javascript' });
const workerURL = URL.createObjectURL(workerBlob);

var workerProcess;

// create input data incl tensor array, return WxH
export async function prepareInputFromFile(nativeFIle, model, blob) {
	return new Promise((resolve, reject) => {
		const context = blob == null ? nativeFIle : blob;
		const reader = new FileReader();

		reader.onload = function (e) {
			const img = new Image();
			img.src = blob == null ? e.target.result : URL.createObjectURL(blob);

			img.onload = async function () {

				const inputData = await imgHelperThreadRun({
					img: nativeFIle,
					w: img.width,
					h: img.height,
					chunkSize: cfg.avgChunkSize,
					layout: model.layout,
					dataType: model.dataType,
					modelChannels: model.channel,
				}, 'decode-transpose');

				inputData.forEach(e => d_in.push(e));
				inputData.length = 0;
				resolve(d_in.length);
			};
			img.onerror = (error) => {
				console.error("error reading file");
				reject(error);
			};
		};
		reader.readAsDataURL(context);
	});
}

// Create input data + tensors from raw pixels (eg: external decoder)
export async function prepareInputFromPixels(img = new Uint8Array(0), width = 0, height = 0, model) {
	// decode-transpose input into float32 pixel data
	const inputData = await imgHelperThreadRun({
		img: img,
		w: width,
		h: height,
		layout: model.layout,
		dataType: model.dataType,
		modelChannels: model.channel,
		chunkSize: cfg.avgChunkSize,
	}, 'transpose-pixels');

	inputData.forEach(e => d_in.push(e));
	inputData.length = 0;
	return d_in.length;
}

export function prepareInputFromTensor(input, width, height, model) {
	d_in.w = width;
	d_in.h = height;
	d_in.c = model.channel;
	d_in.dims = model.layout == 'NCHW' ? [1, model.channel, height, width] : [1, height, width, model.channel];
	d_in.tensor = input.tensor;
}

// set thumbnail for left image (preview)
export function imgUrlFromFile(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = async function (e) {
			const blob = new Blob([reader.result], { type: file.type });
			const objectURL = URL.createObjectURL(blob);
			resolve(objectURL);
		};
		reader.readAsDataURL(file);
	});
}

export function concatUint8Arrays(...arrays) {
	let totalLength = arrays.reduce((acc, array) => acc + array.length, 0);
	let result = new Uint8Array(totalLength);
	let offset = 0;

	arrays.forEach(array => {
		result.set(array, offset);
		offset += array.length;
	});

	return result;
}

/** 
 * Merge two Float32Array NCHW tensors of varying heights into one along the height dimension. 
 * @returns {Float32Array} Merged tensor.
 */

function mergeNCHW(tensor1, tensor2, height1, height2, width, channel = 3, type = 'float32') {
	// Calculate the merged height
	const mergedHeight = height1 + height2;

	const mergedTensor = TypedArray(type, channel * mergedHeight * width);

	for (let c = 0; c < channel; c++) {
		for (let h = 0; h < height1; h++) {
			for (let w = 0; w < width; w++) {
				mergedTensor[c * mergedHeight * width + h * width + w] = tensor1[c * height1 * width + h * width + w];
			}
		}
	}
	for (let c = 0; c < channel; c++) {
		for (let h = 0; h < height2; h++) {
			for (let w = 0; w < width; w++) {
				mergedTensor[c * mergedHeight * width + (h + height1) * width + w] = tensor2[c * height2 * width + h * width + w];
			}
		}
	}

	return mergedTensor;
}

export const mergeTensors = {
	NCHW: mergeNCHW,
}


// Create obj url from RGB for preview
export async function imgUrlFromRGB(array, width, height) {

	if (array.length !== width * height * 4) {
		console.error("Array length does not match dimensions");
		return;
	}

	const offscreenCanvas = new OffscreenCanvas(width, height);
	const context = offscreenCanvas.getContext('2d');
	const imageData = new ImageData(new Uint8ClampedArray(array), width, height);
	context.putImageData(imageData, 0, 0);
	const blob = await offscreenCanvas.convertToBlob();
	const objectURL = URL.createObjectURL(blob);
	return objectURL;
}

// Encode to image format with Canvas API
export async function encodeRGBA(pixels = new Uint8ClampedArray(0), w, h, quality = 100, format = 'jpeg') {
	const blob = await imgHelperThreadRun({
		data: pixels,
		w: w,
		h: h,
		q: quality,
		f: format,
	}, 'encode-pixels');

	return blob;
}

export async function tensorToRGB16_NCHW(tensor, dims) {
	const rgb16 = imgHelperThreadRun({ tensor: tensor, dims: dims }, 'tensor-to-rgb16');
	return rgb16;
}

async function imgHelperThreadRun(input = {}, context) { // resize version 
	return new Promise(async (resolve, reject) => {
		try {
			workerProcess = new Worker(workerURL);
			workerProcess.onmessage = (event) => {
				if (event.data.error) {
					reject(event.data.error);
					return;
				}
				resolve(event.data);
				workerProcess.terminate();
				return;
			};
			workerProcess.onerror = (error) => {
				console.error(error);
				workerProcess.terminate();
				reject(error);
			};
			workerProcess.postMessage({ input: input, context: context });

		} catch (e) {
			reject(e);
		}

	});
}