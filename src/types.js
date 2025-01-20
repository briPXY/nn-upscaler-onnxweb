import { resizeRGBACanvas, insertVertical, insertNewTile, mergeTensorsVertical, extractCanvasData, createCanvas } from "./image_helper";

export class OutputData {
    includeTensor = true;
    preserveAlpha = true;
    dumpOriginalImageData = true;

    #locked = false;
    _insertMethod = "insertVertical";

    multiplier;
    model;

    _tileDim = { x: 0, y: 0, size: 0, resultSize: 0 };
    _tilePos = { x: 0, y: 0 };
    _tempCanvas = null;

    tensorDims;
    tensor;
    imageData = {
        data: new Uint8Array(),
        width: 0,
        height: 0,
    };

    prevData = { w: 0, h: 0, data: null, channels: 0, resized: null }; // Original data, post-padding.
    _prePadding = { w: 0, h: 0, data: null } // Pre-padding original dimensions for cropping 

    constructor({ includeTensor = true, preserveAlpha = true, dumpOriginalImageData = false } = {}) {
        this.includeTensor = includeTensor;
        this.preserveAlpha = preserveAlpha; // Only if model require 3 channels and input has 4.
        this.dumpOriginalImageData = dumpOriginalImageData;
    }

    /**
     * @param {string} v
     */
    set insert(v) {
        this._insertMethod = v;
    }

    /**
     * Operations after full-image/chunks inferences 
     * @param {boolean} cond
     */
    finish() {
        if (this.#locked) {
            return;
        }

        const originalImage = this._prePadding.data ? this._prePadding : this.prevData;
        this.multiplier = this.imageData.width / this.prevData.w;

        if (this._tempCanvas) { // Tiled model
            this.multiplier = this._tileDim.resultSize / this._tileDim.size;
            const finalImg = extractCanvasData(this._tempCanvas, 0, 0, this._prePadding.w * this.multiplier, this._prePadding.h * this.multiplier);
            this.imageData.data = finalImg.data;
            this.imageData.width = finalImg.w;
            this.imageData.height = finalImg.h;
        }

        if (this.multiplier == 1 && this.preserveAlpha && this.prevData.channels == 4) {
            this.#replaceAlpha();
        }
        else if (this.multiplier > 1 && this.preserveAlpha && this.prevData.channels == 4) {
            this.prevData.resized = resizeRGBACanvas(originalImage.data, originalImage.w, originalImage.h, this.multiplier);
            this.#replaceAlpha();
        }

        if (this.dumpOriginalImageData) {
            this.prevData.data = 'dumped';
            this.prevData.resized = 'dumped';
            this._prePadding.data = "dumped";
        }

        this.tensorDims = Dims(this.model.layout, { W: this.imageData.width, H: this.imageData.height, C: this.model.channel, N: 1 });
        this.#locked = true;
    }

    // Merge result pixel data. Called after inference on one chunk
    insertImageChunk(newData) {
        if (this._insertMethod == "insertVertical") {
            insertVertical(newData, this.imageData);
        }
        else {
            if (!this._tempCanvas) {
                this._tempCanvas = createCanvas(this._tileDim.x * newData.width, this._tileDim.y * newData.height);
            }
            insertNewTile(newData, this._tempCanvas, this._tilePos, this._tileDim);
        }
    };

    insertTensorChunk(newTensor, newWidth, newHeight, model) {
        if (this._insertMethod == "insertVertical") {
            this.tensor = mergeTensorsVertical[model.layout](this.tensor, newTensor, this.imageData.height, newHeight, this.imageData.width, model.channel, model.dataType);
        }
    }

    #replaceAlpha() {
        for (let i = 3; i < this.imageData.data.length; i += 4) {
            this.imageData.data[i] = this.prevData.resized[i];
        }
    };

}

export const TypedArray = {
    'float32': (size) => { return new Float32Array(size); },
    'float64': (size) => { return new Float64Array(size); },
    'int8': (size) => { return new Int8Array(size); },
    'uint8': (size) => { return new Uint8Array(size); },
    'float16': (size) => { return new Float32Array(size); },
    'uint16': (size) => { return new Uint16Array(size); },
    'int32': (size) => { return new Int32Array(size); },
    'uint32': (size) => { return new Uint32Array(size); },
    default: (size) => { return new Float32Array(size); },
}

// Return image-type tensor's dims with any layout.
export function Dims(layout, { W = 0, H = 0, C = 3, N = 1 }) {
    layout = layout.toUpperCase();
    const dims = [0, 0, 0, 0];
    dims[layout.indexOf("N")] = N;
    dims[layout.indexOf("C")] = C;
    dims[layout.indexOf("H")] = H;
    dims[layout.indexOf("W")] = W;

    return dims;
}


export const ChunkLevel = {
    1: 40000,
    2: 160000,
    3: 640000,
    4: 1440000
}


export class Model {
    constructor(input) {
        if (typeof input == 'string') {
            this.url = input;
        }

        if (typeof input == 'object' && input.url && input.dataType) {
            for (const prop in input) {
                this.hasOwnProperty(prop) ? this[prop] = input[prop] : null;
            }
        }

        if (input instanceof File) {
            this.url = URL.createObjectURL(input);
        }
    }

    url;
    channel;
    dataType;
    layout;
    tileSize = null;

    validate() {
        const emptyprop = [];
        for (const key in this) {
            if (!this[key] && key != "tileSize") {
                emptyprop.push(key);
            }
        }
        if (emptyprop.length > 0) {
            throw `Model instance is missing information: ${emptyprop}`;
        }
    }
}