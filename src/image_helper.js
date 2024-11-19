import { d_in } from './main'
import workerCodeRawString from './image_helper.worker.js';

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
				const result = { w: img.width, h: img.height };
				// feed to d_in
				d_in.w = img.width;
				d_in.h = img.height;
				d_in.c = model.channel; // most upscale model need 3 channel (?)
				d_in.file = nativeFIle;
				d_in.dims = model.layout == 'NCHW' ? [1, model.channel, img.height, img.width] : [1, img.height, img.width, model.channel];

				const pixelAsTensor = await imgHelperThreadRun({
					img: nativeFIle,
					w: d_in.w,
					h: d_in.h,
					layout: model.layout,
					dataType: model.dataType,
					modelChannels: model.channel,
				}, 'decode-transpose');

				workerProcess.terminate();
				d_in.tensor = pixelAsTensor;
				resolve(result);
			};
			img.onerror = (error) => {
				console.log("error reading file");
				reject(error);
			};
		};
		reader.readAsDataURL(context);
	});
}

// Create input data + tensors from raw pixels (eg: external decoder)
export async function prepareInputFromPixels(img = new Uint8Array(0), width = 0, height = 0, model) {
	// decode-transpose input into float32 pixel data
	const pixelAsTensor = await imgHelperThreadRun({
		img: img,
		w: width,
		h: height,
		layout: model.layout,
		dataType: model.dataType,
		modelChannels: model.channel,
	}, 'transpose-pixels', workerURL);

	workerProcess.terminate();
	d_in.w = width;
	d_in.h = height;
	d_in.c = model.channel;
	d_in.dims = model.layout == 'NCHW' ? [1, model.channel, height, width] : [1, height, width, model.channel];
	d_in.tensor = pixelAsTensor;
	return { w: width, h: height };
}

export function prepareInputFromTensor(input, width, height, model){
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
	const rgb16 = imgHelperThreadRun({tensor: tensor, dims: dims}, 'tensor-to-rgb16');
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
				return;
			};
			workerProcess.onerror = (error) => {
				console.error(error);
				reject(error);
			};
			workerProcess.postMessage({ input: input, context: context });

		} catch (e) {
			reject(e);
		}

	});
}