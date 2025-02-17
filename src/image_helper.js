import { d_in } from './main';
import { cfg } from './cfg.js';
import workerCodeRawString from './image_helper.worker.js';
import { TypedArray } from './types.js';

// Create the Blob and Worker
const workerBlob = new Blob([workerCodeRawString], { type: 'application/javascript' });
const workerURL = URL.createObjectURL(workerBlob);

var workerProcess;

// create tensors from a file input
export async function prepareInputFromFile(nativeFIle, model, blob) {
	return new Promise((resolve, reject) => {
		const context = blob == null ? nativeFIle : blob;
		const reader = new FileReader();

		reader.onload = function (e) {
			const img = new Image();
			img.src = blob == null ? e.target.result : URL.createObjectURL(blob);

			img.onload = async function () {

				const result = await imgHelperThreadRun({
					img: nativeFIle,
					w: img.width,
					h: img.height,
					chunkSize: cfg.avgChunkSize,
					model: model,
				}, 'decode-transpose');

				result.pixelChunks.forEach(e => d_in.push(e));
				result.pixelChunks.length = 0;
				resolve({ totalChunks: d_in.length, prevData: result.prevData, insert: result.insert, tileDim: result.tileDim, prePadding: result.prePadding });
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
	const result = await imgHelperThreadRun({
		img: img,
		w: width,
		h: height,
		model: model,
		chunkSize: cfg.avgChunkSize,
	}, 'transpose-pixels');

	result.pixelChunks.forEach(e => d_in.push(e));
	result.pixelChunks.length = 0;
	return { totalChunks: d_in.length, prevData: result.prevData, insert: result.insert, tileDim: result.tileDim, prePadding: result.prePadding };
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
	return new Promise((resolve) => {
		const reader = new FileReader();
		reader.onload = async function () {
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
 * Merge two typed-array NCHW puput tensors of varying heights into one along the height dimension. 
 * @param {TypedArray} tensor1 - Target tensor
 * @param {TypedArray} tensor2 - New tensor to be merged
 * @param {number} height1 - Target height
 * @param {number} height2 - New height
 * @returns {TypedArray} Merged tensor
 */
function mergeNCHW(tensor1, tensor2, height1, height2, width, channel = 3, type = 'float32') {
	// Calculate the merged height
	const mergedHeight = height1 + height2;

	const mergedTensor = TypedArray[type](channel * mergedHeight * width);

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

export const mergeTensorsVertical = {
	NCHW: mergeNCHW,
};


/**
 * Function to create an ImageBitmap from RGB data
 * @param {number} width - The width of the image
 * @param {number} height - The height of the image
 * @param {Uint8Array} rgbData - The RGB data as a Uint8Array
 * @returns {Promise<ImageBitmap>} - Returns a promise that resolves to an ImageBitmap
 */
export async function createImageBitmapFromRGB(width, height, rgbData) {
	const noAlpha = rgbData.length == width * height * 3;
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');

	const imageData = ctx.createImageData(width, height);

	if (noAlpha) {
		for (let i = 0; i < rgbData.length; i += 3) {
			imageData.data[i] = rgbData[i];       // Red
			imageData.data[i + 1] = rgbData[i + 1]; // Green
			imageData.data[i + 2] = rgbData[i + 2]; // Blue
			imageData.data[i + 3] = 255;           // add Alpha (opaque)
		}
	}
	else {
		imageData.data.set(rgbData);
	}

	ctx.putImageData(imageData, 0, 0);
	const imageBitmap = await createImageBitmap(imageData);

	return imageBitmap;
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

export function stashAlpha(source, dest) {
	for (let i = 3; i < source.length; i += 4) {
		dest.push(source[i]);
	}
}

/**
 * Insert new image output from top-down.
 * @param {Uint8Array} newData - New image
 * @param {Uint8Array} imageData - Target image
 * @returns {Uint8Array} - Merged image
 */
export function insertVertical(newData, imageData) {
	const arrays = [imageData.data, newData.data];
	const totalLength = arrays.reduce((acc, array) => acc + array.length, 0);
	let result = new Uint8Array(totalLength);
	let offset = 0;

	arrays.forEach(array => {
		result.set(array, offset);
		offset += array.length;
	});

	imageData.width = newData.width;
	imageData.height += newData.height;
	imageData.data = result;
	arrays.length = 0;
}

export function createCanvas(width, height) {
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	return canvas;
}

export function insertNewTile(newData, tempCanvas, tilePos, tileDim) {
	const newImageData = new ImageData(new Uint8ClampedArray(newData.data.buffer), newData.width, newData.height);
	const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });

	const coordX = tilePos.x * newData.width;
	const coordY = tilePos.y * newData.height;

	ctx.putImageData(newImageData, coordX, coordY);

	tileDim.resultSize = newData.width;
	tilePos.x += 1;

	if (tilePos.x > tileDim.x - 1) {
		tilePos.x = 0;
		tilePos.y += 1;
	}
}

export function extractCanvasData(canvas, x, y, width, height, clear = true) {
	const ctx = canvas.getContext('2d');
	const imageData = ctx.getImageData(x, y, width, height);
	if (clear) {
		ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas data
		canvas = null;
	}
	return { data: new Uint8Array(imageData.data.buffer), w: width, h: height }
}

// Resize original for the source of AI upscaled alpha channel
export function resizeRGBACanvas(rgbaData, width, height, multiplier) {
	const sourceCanvas = document.createElement('canvas');
	const sourceCtx = sourceCanvas.getContext('2d');
	sourceCanvas.width = width;
	sourceCanvas.height = height;

	const imageData = new ImageData(new Uint8ClampedArray(rgbaData), width, height);
	sourceCtx.putImageData(imageData, 0, 0);

	const targetCanvas = document.createElement('canvas');
	const targetCtx = targetCanvas.getContext('2d');
	targetCanvas.width = Math.round(width * multiplier);
	targetCanvas.height = Math.round(height * multiplier);

	// Set image smoothing properties for better alpha quality
	targetCtx.imageSmoothingEnabled = true;
	targetCtx.imageSmoothingQuality = 'high';

	targetCtx.drawImage(sourceCanvas, 0, 0, targetCanvas.width, targetCanvas.height);

	const resizedData = targetCtx.getImageData(0, 0, targetCanvas.width, targetCanvas.height);

	// Convert Uint8ClampedArray to Uint8Array
	return new Uint8Array(resizedData.data);
}

export async function encodeRGBA(pixels = new Uint8ClampedArray(0), w, h, quality = 100, format = 'jpeg') {
	const blob = await imgHelperThreadRun({
		pixels: pixels,
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
	return new Promise((resolve, reject) => {
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