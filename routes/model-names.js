// Read all files from the directory
const fs = require('fs');
const path = require('path');

// const getFiles = async () => { return await fs.promises.readdir('./static/model') };

async function getAllFiles(dir) {
    let results = [];
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });

    for (let entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            // Recursively get files in subdirectory
            const subDirFiles = await getAllFiles(fullPath);
            results = results.concat(subDirFiles);
        } else {
            // It's a file, add to results
            results.push(fullPath);
        }
    }
    return results;
}

var modelsInfo = null;

module.exports = async function (fastify, opts) {
    if (!modelsInfo) {
        modelsInfo = await getAllFiles('./static/model/')
    }
    fastify.get('/get-models-info', async function (request, reply) {
        return reply.send(modelsInfo);
    })
}
