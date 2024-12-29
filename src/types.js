export class OutputData {
    constructor({ includeTensor = true } = {}) {
        this.includeTensor = includeTensor;
    }

    includeTensor = true;

    tensorChunks;
    tensor;
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

export function TypedArray(type, size = 0) {
    switch (type) {
        case 'float32':
            return new Float32Array(size);
        case 'float64':
            return new Float64Array(size);
        case 'int8':
            return new Int8Array(size);
        case 'uint8':
            return new Uint8Array(size);
        case 'float16':
            return new Float32Array(size);
        case 'uint16':
            return new Uint16Array(size);
        case 'int32':
            return new Int32Array(size);
        case 'uint32':
            return new Uint32Array(size);
        default:
            return new Float32Array(size);
    }
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