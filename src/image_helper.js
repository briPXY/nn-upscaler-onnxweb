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
					layout: model.layout,
					dataType: model.dataType,
					modelChannels: model.channel,
				}, 'decode-transpose');

				result.tensorChunks.forEach(e => d_in.push(e));
				result.tensorChunks.length = 0;
				resolve({ totalChunks: d_in.length, alphaData: result.alphaData, prevData: result.prevData});
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
		layout: model.layout,
		dataType: model.dataType,
		modelChannels: model.channel,
		chunkSize: cfg.avgChunkSize,
	}, 'transpose-pixels');

	result.forEach(e => d_in.push(e));
	result.length = 0;
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

export function stashAlpha(source, dest) {
	for (let i = 3; i < source.length; i += 4) {
		dest.push(source[i]);
	}
}

export function resizeAlphaData(alphaData, originalWidth, originalHeight, multiplier) {
    const newWidth = Math.round(originalWidth * multiplier);
    const newHeight = Math.round(originalHeight * multiplier);
    const resizedAlphaData = new Array(newWidth * newHeight);

    const xRatio = originalWidth / newWidth;
    const yRatio = originalHeight / newHeight;

    for (let newY = 0; newY < newHeight; newY++) {
        for (let newX = 0; newX < newWidth; newX++) {
            // Calculate the position in the original array
            const origX = newX * xRatio;
            const origY = newY * yRatio;

            // Get the four surrounding pixels
            const x1 = Math.floor(origX);
            const y1 = Math.floor(origY);
            const x2 = Math.min(x1 + 1, originalWidth - 1);
            const y2 = Math.min(y1 + 1, originalHeight - 1);
 
            const xWeight = origX - x1;
            const yWeight = origY - y1;
 
            const topLeft = alphaData[y1 * originalWidth + x1];
            const topRight = alphaData[y1 * originalWidth + x2];
            const bottomLeft = alphaData[y2 * originalWidth + x1];
            const bottomRight = alphaData[y2 * originalWidth + x2];

            // Bilinear interpolation
            const top = topLeft * (1 - xWeight) + topRight * xWeight;
            const bottom = bottomLeft * (1 - xWeight) + bottomRight * xWeight;
            const value = top * (1 - yWeight) + bottom * yWeight;

            // Assign the calculated value to the resized array
            resizedAlphaData[newY * newWidth + newX] = Math.round(value);
        }
    }

    return resizedAlphaData;
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