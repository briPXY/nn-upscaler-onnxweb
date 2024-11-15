'use strict'

const path = require('node:path')
const AutoLoad = require('@fastify/autoload')

// Pass --options via CLI arguments in command to enable these options.
const options = {}

module.exports = async function (fastify, opts) {

  fastify.register(require('@fastify/static'), {
    root: path.join(__dirname, 'static'),
    prefix: '/', // optional: default '/'
  })

  // Isolates CORS (required for onnx-wasm SIMD)
  fastify.addHook('onSend', (request, reply, payload, done) => {
    reply.header('Access-Control-Allow-Origin', 'http://127.0.0.1:3000');
    reply.header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
    reply.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    reply.header('Access-Control-Allow-Credentials', 'true');
    reply.header('Cross-Origin-Opener-Policy', 'same-origin');
    reply.header('Cross-Origin-Embedder-Policy', 'require-corp');

    // Proceed to send the response
    done(null, payload);
  });

  fastify.register(AutoLoad, {
    dir: path.join(__dirname, 'plugins'),
    options: Object.assign({}, opts)
  })

  // This loads all plugins defined in routes
  // define your routes in one of these
  fastify.register(AutoLoad, {
    dir: path.join(__dirname, 'routes'),
    options: Object.assign({}, opts)
  })
}

module.exports.options = options
