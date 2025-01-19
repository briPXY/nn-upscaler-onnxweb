class ChunkData {
	constructor({ data, w, h, model } = {}) {
		this.data = data;
		this.w = w;
		this.h = h;
		makeDims(model.layout.toUpperCase(), w, h, model.channel);
	}

	data;
	w;
	h;
	dims;

	makeDims(layout, W, H, C) {
		layout = layout.toUpperCase();
		this.dims = [0, 0, 0, 0];
		dims[layout.indexOf("N")] = 1;
		dims[layout.indexOf("C")] = C;
		dims[layout.indexOf("H")] = H;
		dims[layout.indexOf("W")] = W;
	}
}

//#region data conversion

/* Leverage ort Tensor constructor
// Make pixel array into tensor layout, output is still in 8bit. Channels is input channel (not model).
function transposeToTensor(data, channels = 3, model) {
	const modelAllowAlpha = model.channel == 4;

	if (mode.layout == 'NHWC') {
		if (channels == 4 && !modelAllowAlpha) { // Input 4 channels, model require 3.
			const rgb = [];
			for (let i = 0; i < data.length; i += channels) {
				rgb[i] = data[i];
				rgb[i + 1] = data[i + 1];
				rgb[i + 2] = data[i + 2];
			}
			return rgb;
		}
		if (channels == 3 && modelAllowAlpha) {
			return addAlphaToRGB(data); // Input 3 channels, model require 4.
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

function makeInputTensor(pixels, imgChannels, width, height, model) {
	const tensorArray8bit = transposeToTensor(pixels, imgChannels, model.channel);
	const tensorArrayFloat = createFloatArray(width, height, model);
	convertToFloat(tensorArray8bit, tensorArrayFloat);
	return tensorArrayFloat;
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

*/



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

function makeDims(layout, { W = 0, H = 0, C = 3, N = 1 }) {
	layout = layout.toUpperCase();
	const dims = [0, 0, 0, 0];
	dims[layout.indexOf("N")] = N;
	dims[layout.indexOf("C")] = C;
	dims[layout.indexOf("H")] = H;
	dims[layout.indexOf("W")] = W;

	return dims;
}

function retainAlpha(pixels, input_channels, model_channels) {
	let retainedAlpha = [], i = 0;
	if (input_channels == 4 && model_channels == 3) {
		for (i; i < pixels.length; i += 4) {
			retainedAlpha.push(pixels[i]);
		}
	}
	return retainedAlpha;
}

//#region padding
function calculatePadding(size, tileSize) {
	const remainder = size % tileSize;
	const padding = remainder === 0 ? 0 : tileSize - remainder;
	return padding;
}

function calculatePaddedDimensions(width, height, tileSize) {
	const paddingRight = calculatePadding(width, tileSize);
	const paddingBottom = calculatePadding(height, tileSize);
	const newWidth = width + paddingRight;
	const newHeight = height + paddingBottom;

	return {
		paddingRight,
		paddingBottom,
		width: newWidth,
		height: newHeight,
	};
}


//#endregion
//#region decode to rgb

function drawRGB(blob, width, height) {
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

const loadImage = async (source, width, height) => {
	return new Promise((resolve, reject) => {
		var blob;
		var fileDataArray;
		if (typeof source === 'string') {
			fetch(source)
				.then(response => response.blob())
				.then(async (blob) => {
					let result = await drawRGB(blob, width, height);
					resolve(result);
				})
				.catch(error => reject(error));
		}
		if (typeof source === 'object') {
			const reader = new FileReader();
			reader.onload = async function (e) {
				fileDataArray = new Uint8Array(e.target.result);
				blob = new Blob([fileDataArray], { type: source.type });
				let result = await drawRGB(blob, width, height);
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

// function createFloatArray(width, height, model) {
// 	if (model.dataType == 'float32') {
// 		return new Float32Array(width * height * model.channel);
// 	}
// 	if (model.dataType == 'float16') {
// 		return new Float32Array(width * height * model.channel);
// 	}
// 	if (model.dataType == 'float64') {
// 		return new Float64Array(width * height * model.channel);
// 	}
// }

//#region slicers

function paddedImageData(originalData, originalWidth, originalHeight, newWidth, newHeight) {
	const bytesPerPixel = 3;
	const newSize = newWidth * newHeight * bytesPerPixel;
 
	const paddedData = new Uint8Array(newSize).fill(0);

	// Fill the new array with the original image data
	for (let y = 0; y < originalHeight; y++) {
		for (let x = 0; x < originalWidth; x++) {
			for (let c = 0; c < bytesPerPixel; c++) {
				const originalIndex = (y * originalWidth + x) * bytesPerPixel + c;
				const paddedIndex = (y * newWidth + x) * bytesPerPixel + c;
				paddedData[paddedIndex] = originalData[originalIndex];
			}
		}
	}
 
	return paddedData;
}

// Work-in-Progress
function pixelSlicerSquare(pixels, width, height, model) {
	const tileSize = model.tileSize;
	const channels = input.length == (width * height * 3) ? 3 : 4;
	const retainedAlpha = retainAlpha(pixels, channels, model.channel);

	const newDim = calculatePaddedDimensions(width, height, tileSize);
	const newData = paddedImageData(pixels, width, height, newDim.width, newDim.height);

	const tileX = newDim.width / tileSize;
	const tileY = newDim.height / tileSize;
	const pixelChunks = [];

	for (let y = 0; y < tileY; y++) {
		for (let x = 0; x < tileX; x++) {
			const tileData = [];

			for (let ty = 0; ty < tileSize; ty++) {
				for (let tx = 0; tx < tileSize; tx++) {
					const srcIndex = ((y * tileSize + ty) * width + (x * tileSize + tx)) * 4;
					const destIndex = (ty * tileSize + tx) * 4;

					tileData[destIndex] = newData[srcIndex];       // Red
					tileData[destIndex + 1] = newData[srcIndex + 1]; // Green
					tileData[destIndex + 2] = newData[srcIndex + 2]; // Blue
					tileData[destIndex + 3] = newData[srcIndex + 3]; // Alpha
				}
			}

			pixelChunks.push(new ChunkData({
				data: tileData,
				h: tileSize,
				w: tileSize,
				model: model, 
			}));
		}
	}

	self.postMessage({
		pixelChunks: pixelChunks,
		insert: "insertTiles",
		alphaData: retainedAlpha,
		prevData: { data: newData, w: newDim.width, h: newDim.height, channels: channels },
		tileDim: { x: tileX, y: tileY, size: model.tileSize},
		prePadding: {w: width, h: height},
	});
	pixelChunks = null;
	return;

}

function pixelsSlicerVertical(pixels, chunkSize, width, height, model) {
	let pixelChunks = [];
	const originalHeight = height;

	const channels = pixels.length / (width * height);
	const retainedAlpha = retainAlpha(pixels, channels, model.channel);

	if (width * height <= chunkSize) {
		return [new ChunkData({ data: pixels, h: height, w: width })];
	}

	const chunkHeight = Math.round(chunkSize / width);
	const sliceSize = width * chunkHeight * channels;

	for (let i = 0; i < pixels.length;) {
		if (i + sliceSize >= pixels.length) {
			pixelChunks.push(new ChunkData({ data: pixels.slice(i), h: height, w: width, model: model }));
			break;
		}

		pixelChunks.push(new ChunkData({ data: pixels.slice(i, i + sliceSize), h: chunkHeight, w: width, model: model }));
		height -= chunkHeight;
		i += sliceSize;
	}

	self.postMessage({ 
		pixelChunks: pixelChunks, 
		insert: "insertVertical", 
		alphaData: retainedAlpha, 
		prevData: { data: pixels, w: width, h: originalHeight, channels: channels },
		prePadding: null,
		tileDim: null,
	});
	pixelChunks = null;
	return;
}

//#endregion
//#region message listener

self.onmessage = async function (event) {
	const data = event.data.input;
	const input = data.img; // Array buffer or a file.   

	// Create sliced image from typed array.
	if (event.data.context == 'transpose-pixels') { // Input is rgb/a pixels (uint8). 
		data.model.tileSize ? pixelSlicerSquare(input, data.w, data.h, data.model) : pixelsSlicerVertical(input, data.chunkSize, data.w, data.h, data.model);
	}

	// Create sliced image from image File.
	if (event.data.context == "decode-transpose") { // Input is native image File object.
		loadImage(input, data.w, data.h)
			.then((pixels) => {
				data.model.tileSize ? pixelSlicerSquare(pixels, data.w, data.h, data.model) : pixelsSlicerVertical(pixels, data.chunkSize, data.w, data.h, data.model);
			})
			.catch((error) => {
				console.error(error);
				self.postMessage({ error: error.message });
			});
	}

	// Create image blob from output tensor.
	if (event.data.context == "transpose-encode") {
		try {
			let rgb8 = new Uint8Array(data.w * data.h * data.c);
			convertTo8Bit(data.data, rgb8);
			const outputFile = await createBlobFromRgbData(rgb8, data.w, data.h, data.q / 100, data.c, data.f);
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
			const outputFile = await createBlobFromRgbData(data.pixels, data.w, data.h, data.q / 100, data.f);
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
