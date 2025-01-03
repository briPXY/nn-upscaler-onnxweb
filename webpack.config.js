const path = require('path');
const CopyPlugin = require("copy-webpack-plugin"); 

const all = [
    {
        from: 'node_modules/onnxruntime-web/dist/*.wasm', // Copy all .wasm files
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/*.mjs', // Copy all .mjs files (needed for ES module)
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/*.js', // Copy all .js files (if needed)
        to: '[name][ext]'
    }];

const spec = [
    {
        from: 'node_modules/onnxruntime-web/dist/*.wasm', // Copy all .wasm files
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort.webgpu.min.js',
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort.wasm.min.js',
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort.all.min.js',
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort.bundle.min.mjs',
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs',
        to: '[name][ext]'
    },
    {
        from: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs',
        to: '[name][ext]'
    },
];

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, 'src', 'main.js'),
        output: {
            path: path.resolve(__dirname, 'static', 'onnxruntime-web', 'dist'),
            filename: 'bundle.min.js',
            library: {
                name: 'wnx',
                type: 'umd'
            }
        },
        module: {
            rules: [
                {
                    test: /\.worker\.js$/,
                    use: 'raw-loader',
                },
            ],
        },
        plugins: [new CopyPlugin({
            patterns: spec,
        })],
        mode: 'production'
    }
};