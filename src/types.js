import { resizeRGBACanvas, mergeVertical, insertNewTile, mergeTensorsVertical } from "./image_helper";

export class OutputData {
    includeTensor = true;
    preserveAlpha = true;
    dumpOriginalImageData = false;

    #locked = false;
    _insert = "insertVertical";

    multiplier;
    _tileDim;
    #tilePos = { x: 0, y: 0 };

    tensor;
    imageData = {
        data: new Uint8Array(),
        width: 0,
        height: 0,
    };

    alphaData = [];

    prevData = {
        data: null,
        resized: null,
    };

    _prePadding = {} // Pre-padding original dimensions for cropping

    constructor({ includeTensor = true, preserveAlpha = true, dumpOriginalImageData = false } = {}) {
        this.includeTensor = includeTensor;
        this.preserveAlpha = preserveAlpha; // Only if model require 3 channels and input has 4.
        this.dumpOriginalImageData = dumpOriginalImageData;
    }

    /**
     * @param {string} v
     */
    set insert(v) {
        this._insert = '_' + v;
    }

    /**
     * Operations after full-image/chunks inferences 
     * @param {boolean} cond
     */
    set finish(cond) {
        if (!cond || this.#locked) {
            return;
        }

        this.multiplier = this.imageData.width / this.prevData.w;

        if (this.multiplier == 1 && this.alphaData.length > 0 && this.preserveAlpha) {
            this.#replaceAlpha();
        }
        else if (this.multiplier > 1 && this.alphaData.length > 0 && this.preserveAlpha) {
            this.prevData.resized = resizeRGBACanvas(this.prevData.data, this.prevData.w, this.prevData.h, this.multiplier);
            this.#replaceAlpha();
        }

        if (this.dumpOriginalImageData) {
            this.prevData.data = null;
            this.prevData.resized = null;
        }
        
        this.#locked = true;
    }

    _insertVertical(newData) {
        mergeVertical(newData, this.imageData);
    };

    _insertTile(newData) {
        insertNewTile(newData, this.imageData, pos);
    } 

    // Merge result pixel data. Called after inference on one chunk
    insertImageChunk(data) {
        this[this._insert](data); // insertVertical() or insertTile()
    };

    insertTensorChunk(newTensor, newWidth, newHeight, model){
        if (this._insert == "insertVertical"){
            this.tensor = mergeTensorsVertical[model.layout](this.tensor, newTensor, this.imageData.height, newHeight, this.imageData.width, model.channel, model.dataType);
        } 
    }

    #replaceAlpha() {
        for (let i = 0; i < this.imageData.data.length; i += 4) {
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