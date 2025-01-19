import { resizeRGBACanvas } from "./image_helper";

export class OutputData {
    includeTensor = true;
    preserveAlpha = true;
    dumpOriginalImageData = false;

    #locked = false;
    _havePadding = false;
    _insert = "insertVertical";

    tensorChunks;
    multiplier;

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

    constructor({ includeTensor = true, preserveAlpha = true, dumpOriginalImageData = false } = {}) {
        this.includeTensor = includeTensor;
        this.preserveAlpha = preserveAlpha; // Only if model require 3 channels and input has 4.
        this.dumpOriginalImageData = dumpOriginalImageData;
    }

    /**
     * @param {string} v
     */
    set insert(v) {
        this._insert = v;
    }

    get tensor() {
        return this._tensor;
    }
    /**
     * @param {TypedArray} value
     */
    set tensor(value) {
        if (this.includeTensor) {
            this._tensor = value;
        }
    }

    /**
     * @param {boolean} cond
     */
    set _finish(cond) {
        if (!cond || !this.preserveAlpha || this.#locked) {
            return;
        }
        if (this.multiplier == 1 && this.alphaData.length > 0 && this.preserveAlpha) {
            this.replaceAlpha();
        }
        if (this.multiplier > 1 && this.alphaData.length > 0 && this.preserveAlpha) {
            this.prevData.resized = resizeRGBACanvas(this.prevData.data, this.prevData.w, this.prevData.h, this.multiplier);
            this.replaceAlpha();
        }
        if (this.dumpOriginalImageData){
            this.prevData.data = null;
            this.prevData.resized = null;
        }
        this.#locked = true;
    }

    insertVertical(imageData) {
        const arrays = [this.imageData.data, imageData.data];
        const totalLength = arrays.reduce((acc, array) => acc + array.length, 0);
        let result = new Uint8Array(totalLength);
        let offset = 0;

        arrays.forEach(array => {
            result.set(array, offset);
            offset += array.length;
        });

        arrays.length = 0
        this.imageData.width = imageData.width;
        this.imageData.height += imageData.height;
        this.imageData.data = result;
    };

    insertImageChunk(data) {
        this[this._insert](data); // insertVertical() or insertTile()
    };

    replaceAlpha() {
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
    tileSize;

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