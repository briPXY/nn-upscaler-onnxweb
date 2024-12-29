
var width = 0;
var height = 0;
var stripAlpha = false;
var tensorLayout = 'NCHW';
var dataType = 'float32';
//#region data conversion

// Make pixel array into tensor layout, output is still in 8bit. Channels is input channel (not model).
function transposeToTensor(data, channels = 3, modelChannels) {
	const modelAllowAlpha = modelChannels == 4;

	if (tensorLayout == 'NHWC') {
		if (channels == 4 && !modelAllowAlpha) { // Input is 4 channels and model is 3 channels.
			const rgb = [];
			for (let i = 0; i < data.length; i += channels) {
				rgb[i] = data[i];
				rgb[i + 1] = data[i + 1];
				rgb[i + 2] = data[i + 2];
				// Skip alpha.
			}
			return rgb;
		}
		if (channels == 3 && modelAllowAlpha) {
			return addAlphaToRGB(data); // Input 3 channels, model need 4.
		}
		else {
			return data;
		}
	}
	else { // for NCHW
		const [R, G, B, A] = [[], [], [], []];
		const addAlpha = modelAllowAlpha && channels == 3;
		const includeAlpha = modelAllowAlpha && channels == 4;

		for (let i = 0; i < data.length; i += channels) {
			R.push(data[i]);
			G.push(data[i + 1]);
			B.push(data[i + 2]);
			addAlpha ? A.push(255) : null;
			includeAlpha ? A.push(data[i + 3]) : null;
			// Alpha skipped if model is 3 and input is 4 channels.
		}

		// 1b. concatenate RGB ~= transpose [224, 224, 3] -> [3, 224, 224] 
		const nchwData = modelAllowAlpha ? R.concat(G).concat(B).concat(A) : R.concat(G).concat(B);
		return nchwData;
	}
}

function tensorToRGB(tensorData, width, height, channels) { // For NCHW tensor output layout
	const data = new Uint8Array(width * height * channels);
	const numPixels = width * height;

	// Each channel's data is stored sequentially in transposedData
	const R = tensorData.slice(0, numPixels);
	const G = tensorData.slice(numPixels, numPixels * 2);
	const B = tensorData.slice(numPixels * 2, numPixels * 3);

	if (channels == 3) {
		for (let i = 0; i < numPixels; i++) {
			data[i * 4] = R[i];        // Red
			data[i * 4 + 1] = G[i];    // Green
			data[i * 4 + 2] = B[i];    // Blue 
		}
	}
	else {
		for (let i = 0; i < numPixels; i++) {
			data[i * 4] = R[i];        // Red
			data[i * 4 + 1] = G[i];    // Green
			data[i * 4 + 2] = B[i];    // Blue 
			data[i * 4 + 3] = 255;
		}
	}

	return data;
}

function convertToFloat(transposed, transposed32) { // All types of floats
	let i, l = transposed.length;
	for (i = 0; i < l; i++) {
		transposed32[i] = transposed[i] / 255.0;
	}
}

function convertTo8Bit(transposed32, tensorArray8bit) {
	let i, l = transposed32.length; // length of the input float array
	for (i = 0; i < l; i++) {
		tensorArray8bit[i] = Math.round(transposed32[i] * 255); // convert back to 8bit and round
	}
}

function addAlphaToRGB(rgbArray, alphaValue = 255) {
	if (rgbArray.length % 3 !== 0) {
		throw new Error("Invalid RGB array length");
	}
	const numPixels = rgbArray.length / 3;
	const rgbaArray = new Uint8Array(numPixels * 4);
	for (let i = 0, j = 0; i < rgbArray.length; i += 3, j += 4) {
		rgbaArray[j] = rgbArray[i];       // Red
		rgbaArray[j + 1] = rgbArray[i + 1]; // Green
		rgbaArray[j + 2] = rgbArray[i + 2]; // Blue
		rgbaArray[j + 3] = alphaValue;     // Alpha
	}
	return rgbaArray;
}

// Convert tensor NCHW to 16 bit rgba
function tensorNCHW_to_RGB16(tensor, dims) {
	const batch = dims[0];
	const channels = dims[1]; // RGB
	const height = dims[2];
	const width = dims[3];

	const rgba16bit = new Uint16Array(batch * height * width * channels);

	// Rearrange NCHW to NHWC and scale to uint16 range
	let index = 0;
	for (let b = 0; b < batch; b++) {
		for (let h = 0; h < height; h++) {
			for (let w = 0; w < width; w++) {
				for (let c = 0; c < channels; c++) {
					// Calculate NCHW index
					const nchwIndex = b * (channels * height * width) +
						c * (height * width) +
						h * width +
						w;

					rgba16bit[index++] = Math.max(0, Math.min(65535, Math.round(tensor[nchwIndex] * 65535)));
				}
			}
		}
	}
	return rgba16bit;
}

function makeInputTensor(pixels, imgChannels, width, height, modelChannels) {
	const tensorArray8bit = transposeToTensor(pixels, imgChannels, modelChannels);
	const tensorArrayFloat = createFloatArray(width, height, modelChannels);
	convertToFloat(tensorArray8bit, tensorArrayFloat);
	return tensorArrayFloat;
}

function makeDims(layout, { W = 0, H = 0, C = 3, N = 1 }) {
	layout = layout.toUpperCase();
	const dims = [0, 0, 0, 0];
	dims[layout.indexOf("N")] = N;
	dims[layout.indexOf("C")] = C;
	dims[layout.indexOf("H")] = H;
	dims[layout.indexOf("W")] = W;

	return dims;
}

//#endregion
//#region decode to rgb

function drawRGB(blob) {
	return new Promise(async (resolve, reject) => {
		createImageBitmap(blob).then(
			function (bitmap) {
				const offscreenCanvas = new OffscreenCanvas(width, height);
				const context = offscreenCanvas.getContext('2d');
				context.drawImage(bitmap, 0, 0, width, height);
				const imageData = context.getImageData(0, 0, width, height);
				resolve(imageData.data);
			})
	});
}

const loadImage = async (source) => {
	return new Promise((resolve, reject) => {
		var blob;
		var fileDataArray;
		if (typeof source === 'string') {
			fetch(source)
				.then(response => response.blob())
				.then(async (blob) => {
					let result = await drawRGB(blob);
					resolve(result);
				})
				.catch(error => reject(error));
		}
		if (typeof source === 'object') {
			const reader = new FileReader();
			reader.onload = async function (e) {
				fileDataArray = new Uint8Array(e.target.result);
				blob = new Blob([fileDataArray], { type: source.type });
				let result = await drawRGB(blob);
				resolve(result);
			};
			reader.readAsArrayBuffer(source);
		}
	});
}

//#endregion
//#region encode to format

function drawRectangleWithColor(offscreenCanvas, rgbArray) {
	const offscreenContext = offscreenCanvas.getContext('2d');
	const imageData = offscreenContext.createImageData(offscreenCanvas.width, offscreenCanvas.height);
	imageData.data.set(rgbArray);
	offscreenContext.putImageData(imageData, 0, 0);
	return;
}

async function fileFromOffscreenCanvas(offscreenCanvas, rgb, quality, format) {
	const offscreenContext = offscreenCanvas.getContext('2d');

	drawRectangleWithColor(offscreenCanvas, rgb);
	rgb = null;
	// Get the image data
	const imageData = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
	const tempCanvas = new OffscreenCanvas(offscreenCanvas.width, offscreenCanvas.height);
	const tempContext = tempCanvas.getContext('2d');
	tempContext.putImageData(imageData, 0, 0);

	// Convert the canvas to image
	format == "jpg" ? format = "jpeg" : format;
	const fileblob = await tempCanvas.convertToBlob({ type: `image/${format.toLowerCase()}`, quality: quality });
	return fileblob;
}

async function createBlobFromRgbData(pixelData, width, height, compressionQuality = 0.75, format) {
	return new Promise(async (resolve, reject) => {
		if (pixelData.length != width * height * 4) {
			pixelData = addAlphaToRGB(pixelData);
		}
		const offscreenCanvas = new OffscreenCanvas(width, height);
		fileFromOffscreenCanvas(offscreenCanvas, pixelData, compressionQuality, format)
			.then(async (blob) => {
				// Convert raw pixel data to PNG image
				resolve(blob);
			})
			.catch(error => { console.error(error); reject(error); });
	})
}

function createFloatArray(width, height, modelChannels) {
	if (dataType == 'float32') {
		return new Float32Array(width * height * modelChannels);
	}
	if (dataType == 'float16') {
		return new Float32Array(width * height * modelChannels);
	}
	if (dataType == 'float64') {
		return new Float64Array(width * height * modelChannels);
	}
}

//#region create input chunks

function pixelsSlicer(pixels, chunkSize, width, height, channels) {
	const pixelChunks = [];

	if (width * height <= chunkSize) {
		return [{ data: pixels, h: height }];
	}

	const chunkHeight = Math.round(chunkSize / width);
	const sliceSize = width * chunkHeight * channels;

	for (let i = 0; i < pixels.length;) {
		if (i + sliceSize >= pixels.length) {
			pixelChunks.push({ data: pixels.slice(i), h: height });
			break;
		}

		pixelChunks.push({ data: pixels.slice(i, i + sliceSize), h: chunkHeight });
		height -= chunkHeight;
		i += sliceSize;
	}

	return pixelChunks;
}

//#endregion
//#region main-thread listener

self.onmessage = async function (event) {
	const data = event.data.input;
	const input = data.img; // Array buffer or a file.
	width = data.w;
	height = data.h;
	tensorLayout = data.layout;
	dataType = data.dataType;

	// Create tensor array from pixels.
	if (event.data.context == 'transpose-pixels') { // Input is rgb/a pixels (uint8).
		const channels = input.length == (width * height * 3) ? 3 : 4;
		let tensorChunks = [];
		const pixelChunks = pixelsSlicer(input, data.chunkSize, width, height, channels);

		for (let i = 0; i < pixelChunks.length; i++) {
			const tensorArrayFloat = makeInputTensor(pixelChunks[i].data, channels, width, pixelChunks[i].h, data.modelChannels);
			const dims = makeDims(tensorLayout, { C: data.modelChannels, N: 1, W: width, H: pixelChunks[i].h });
			tensorChunks.push({ tensor: tensorArrayFloat, dims: dims, w: width, h: pixelChunks[i].h, c: data.modelChannels });
		}

		self.postMessage(tensorChunks);
		tensorChunks = null;
		return;
	}

	// Create tensor array from image File.
	if (event.data.context == "decode-transpose") { // Input is native image File object.
		loadImage(input)
			.then((pixels) => {
				let tensorChunks = [];
				const channels = input.length == (width * height * 3) ? 3 : 4;
				const pixelChunks = pixelsSlicer(pixels, data.chunkSize, width, height, channels);

				for (let i = 0; i < pixelChunks.length; i++) {
					const tensorArrayFloat = makeInputTensor(pixelChunks[i].data, channels, width, pixelChunks[i].h, data.modelChannels);
					const dims = makeDims(tensorLayout, { C: data.modelChannels, N: 1, W: width, H: pixelChunks[i].h });
					tensorChunks.push({ tensor: tensorArrayFloat, dims: dims, w: width, h: pixelChunks[i].h, c: data.modelChannels });
				}
				// Send the processed data back to the main threa
				self.postMessage(tensorChunks);
				tensorChunks = null;
				return;
			})
			.catch((error) => {
				console.error(error);
				self.postMessage({ error: error.message });
			});
	}

	// Create image blob from output tensor.
	if (event.data.context == "transpose-encode") {
		try {
			let rgb8 = new Uint8Array(width * height * data.c);
			convertTo8Bit(data.data, rgb8);
			const outputFile = await createBlobFromRgbData(rgb8, width, height, data.q / 100, data.c, data.f);
			self.postMessage(outputFile);
		}
		catch (error) {
			console.error(error);
			self.postMessage({ error: error.message });
		}
	}

	// Create image blob from uint8 pixel data.
	if (event.data.context == "encode-pixels") {
		try {
			const outputFile = await createBlobFromRgbData(data.data, width, height, data.q / 100, data.f);
			self.postMessage(outputFile);
		}
		catch (error) {
			console.error(error);
			self.postMessage({ error: error.message });
		}
	}

	if (event.data.context == 'tensor-to-rgb16') {
		const rgb16 = tensorNCHW_to_RGB16(data.tensor, data.dims);
		self.postMessage(rgb16);
	}
};
