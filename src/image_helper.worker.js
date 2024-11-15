
var width = 0;
var height = 0;
var stripAlpha = false;
var tensorLayout = 'NCHW';
var dataType = 'float32';
//#region data conversion

// Make pixel array into tensor layout, output is still in 8bit.
function transposeToTensor(data) {
	const [R, G, B] = [[], [], []]
	for (let i = 0; i < data.length; i += 4) {
		R.push(data[i]);
		G.push(data[i + 1]);
		B.push(data[i + 2]);
		// 2. skip data[i + 3] thus filtering out the alpha channel
	}

	if (tensorLayout == 'NHWC') {
		const nhwcData = [];
		// Reshape into NHWC format: [height, width, 3]
		for (let i = 0; i < height * width; i++) {
			nhwcData[i * 3] = R[i];
			nhwcData[i * 3 + 1] = G[i];
			nhwcData[i * 3 + 2] = B[i];
		}
		return nhwcData;
	}
	else {
		// 1b. concatenate RGB ~= transpose [224, 224, 3] -> [3, 224, 224]
		const nchwData = R.concat(G).concat(B);
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

function createFloatArray(size) {
	if (dataType == 'float32') {
		return new Float32Array(size);
	}
	if (dataType == 'float16') {
		return new Float16Array(size);
	}
	if (dataType == 'float64') {
		return new Float64Array(size);
	}
}

//#endregion
//#region main-thread listener

self.onmessage = async function (event) {
	const input = event.data.input
	const data = input.data; // Image file obj or array of pixels.
	width = input.w;
	height = input.h;
	tensorLayout = input.layout;
	dataType = input.dataType;

	// Create tensor array from pixels.
	if (event.data.context == 'transpose-pixels') { // Input is rgb/a pixels (uint8).
		const tensorArray8bit = transposeToTensor(input);
		const tensorArrayFloat = createFloatArray(width * height * 3);
		convertToFloat(tensorArray8bit, tensorArrayFloat);
		self.postMessage(tensorArrayFloat);
	}

	// Create tensor array from image File.
	if (event.data.context == "decode-transpose") { // Input is native image File object.
		loadImage(data)
			.then((pixels) => {
				const tensorArray8bit = transposeToTensor(pixels);
				const tensorArrayFloat = createFloatArray(width * height * 3);
				convertToFloat(tensorArray8bit, tensorArrayFloat);
				// Send the processed data back to the main threa
				self.postMessage(tensorArrayFloat);
			})
			.catch((error) => {
				console.error(error);
				self.postMessage({ error: error.message });
			});
	}

	// Create image blob from output tensor.
	if (event.data.context == "transpose-encode") {
		try {
			let rgb8 = new Uint8Array(width * height * input.c);
			convertTo8Bit(data, rgb8);
			const outputFile = await createBlobFromRgbData(rgb8, width, height, input.q / 100, input.c, input.f);
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
			const outputFile = await createBlobFromRgbData(data, width, height, input.q / 100, input.f);
			self.postMessage(outputFile);
		}
		catch (error) {
			console.error(error);
			self.postMessage({ error: error.message });
		}
	}

	if (event.data.context == 'tensor-to-rgb16'){
		const rgb16 = tensorNCHW_to_RGB16(input.tensor, input.dims);
		self.postMessage(rgb16);
	}
};