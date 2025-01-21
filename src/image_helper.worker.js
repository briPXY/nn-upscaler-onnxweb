class ChunkData {
	constructor({ data, w, h, model } = {}) {
		this.data = data;
		this.w = w;
		this.h = h;
		this.makeDims(model.layout.toUpperCase(), w, h, model.channel);
	}

	data;
	w;
	h;
	dims;

	makeDims(layout, W, H, C) {
		this.dims = [0, 0, 0, 0];
		this.dims[layout.indexOf("N")] = 1;
		this.dims[layout.indexOf("C")] = C;
		this.dims[layout.indexOf("H")] = H;
		this.dims[layout.indexOf("W")] = W;
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

function makeDims(layout, { W = 0, H = 0, C = 3, N = 1 }) {
	layout = layout.toUpperCase();
	const dims = [0, 0, 0, 0];
	dims[layout.indexOf("N")] = N;
	dims[layout.indexOf("C")] = C;
	dims[layout.indexOf("H")] = H;
	dims[layout.indexOf("W")] = W;

	return dims;
}

function createCanvas(pixels, width, height) {
	const canvas = new OffscreenCanvas(width, height);
	const ctx = canvas.getContext('2d', { willReadFrequently: true });
	const imageData = new ImageData(new Uint8ClampedArray(pixels.buffer), width, height)
	ctx.putImageData(imageData, 0, 0);
	return canvas;
}

function cropCanvasData(canvas, x, y, width, height) {
	const ctx = canvas.getContext('2d');
	const imageData = ctx.getImageData(x, y, width, height);
	return new Uint8Array(imageData.data.buffer);
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

//#region Image slicers

function paddedImageData(originalData, originalWidth, originalHeight, newWidth, newHeight, channels) {
	const newSize = newWidth * newHeight * channels;

	const paddedData = new Uint8Array(newSize).fill(0);

	// Fill the new array with the original image data
	for (let y = 0; y < originalHeight; y++) {
		for (let x = 0; x < originalWidth; x++) {
			for (let c = 0; c < channels; c++) {
				const originalIndex = (y * originalWidth + x) * channels + c;
				const paddedIndex = (y * newWidth + x) * channels + c;
				paddedData[paddedIndex] = originalData[originalIndex];
			}
		}
	}

	return paddedData;
}

// Tile based slicer, from top
function pixelSlicerSquare(pixels, width, height, model) {
	const tileSize = model.tileSize;
	const channels = pixels.length / (width * height);

	const newDim = calculatePaddedDimensions(width, height, tileSize);
	const newData = paddedImageData(pixels, width, height, newDim.width, newDim.height, 4);
	const canvas = createCanvas(newData, newDim.width, newDim.height);

	const tileX = newDim.width / tileSize;
	const tileY = newDim.height / tileSize;
	const totalTiles = tileX * tileY;
	let tx = 0, ty = 0;

	const pixelChunks = [];

	for (let i = 0; i < totalTiles; i++) {
		const croppedData = cropCanvasData(canvas, tx * tileSize, ty * tileSize, tileSize, tileSize);
		tx += 1;

		if (tx >= tileX) {
			tx = 0;
			ty += 1;
		}

		pixelChunks.push(new ChunkData({
			data: croppedData,
			w: tileSize,
			h: tileSize,
			model: model,
		}));

	}

	self.postMessage({
		pixelChunks: pixelChunks,
		insert: "insertTile",
		prevData: { data: newData, w: newDim.width, h: newDim.height, channels: channels },
		tileDim: { x: tileX, y: tileY, size: model.tileSize },
		prePadding: { data: pixels, w: width, h: height },
	});

	canvas.getContext("2d").clearRect(0, 0, newDim.width, newDim.height);
	pixelChunks.length = 0;
	return;

}

function pixelsSlicerVertical(pixels, chunkSize, width, height, model) {
	let pixelChunks = [];
	const originalHeight = height;

	const channels = pixels.length / (width * height);

	if (width * height <= chunkSize) {
		pixelChunks.push(new ChunkData({ data: pixels, h: height, w: width, model: model }));
	}

	else { 
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
	}

	self.postMessage({
		pixelChunks: pixelChunks,
		insert: "insertVertical",
		prevData: { data: pixels, w: width, h: originalHeight, channels: channels },
		prePadding: { w: width, h: originalHeight, data: null },
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
