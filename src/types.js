export class OutputData {
    constructor({ includeTensor = true } = {}) {
        this.includeTensor = includeTensor;
    }

    includeTensor = true;

    tensorChunks;
    multiplier;
    imageData = {
        data: new Uint8Array(),
        width: 0,
        height: 0
    };

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

    validate() {
        const emptyprop = [];
        for (const key in this) {
            if (!this[key]) {
                emptyprop.push(key);
            }
        }
        if (emptyprop.length > 0) {
            throw `Model instance is missing information: ${emptyprop}`;
        }
    }
}